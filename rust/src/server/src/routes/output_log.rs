// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Generated-output logging for `--enable-log-outputs`, mirroring Python's
//! `RequestLogger.log_outputs`.

use tracing::{debug, info};
use vllm_chat::AssistantBlockKind;

use crate::config::OutputLogConfig;

/// Which part of a response one output log line covers.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LoggedOutput {
    /// The complete output of a non-streaming response.
    Response,
    /// One streamed delta.
    StreamDelta,
    /// The accumulated output of a finished stream.
    StreamComplete,
}

impl OutputLogConfig {
    /// Log one generated output like Python's `RequestLogger.log_outputs`: the
    /// output text and finish reason at INFO, the output token IDs at DEBUG.
    /// The request ID comes from the enclosing request span. With
    /// `max_log_len`, only that many leading characters and token IDs are
    /// logged.
    pub(crate) fn log(
        self,
        kind: LoggedOutput,
        output: &str,
        token_ids: Option<&[u32]>,
        finish_reason: Option<&str>,
    ) {
        let stream = match kind {
            LoggedOutput::Response => None,
            LoggedOutput::StreamDelta => Some("delta"),
            LoggedOutput::StreamComplete => Some("complete"),
        };
        if let Some(token_ids) = token_ids {
            let token_ids = match self.max_log_len {
                Some(max_log_len) => &token_ids[..token_ids.len().min(max_log_len)],
                None => token_ids,
            };
            debug!(stream, ?token_ids, "generated response details");
        }
        let output =
            match self.max_log_len.and_then(|max_log_len| output.char_indices().nth(max_log_len)) {
                Some((end, _)) => &output[..end],
                None => output,
            };
        info!(stream, ?output, finish_reason, "generated response");
    }
}

/// Describe a chat output for the log the way Python does: the content, or
/// the tool calls as `[tool_calls: name(arguments), ...]` when there is no
/// content.
pub(crate) fn chat_output_text<'a>(
    content: &str,
    tool_calls: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> String {
    if !content.is_empty() {
        return content.to_string();
    }
    let tool_calls = tool_calls
        .into_iter()
        .map(|(name, arguments)| format!("{name}({arguments})"))
        .collect::<Vec<_>>();
    if tool_calls.is_empty() {
        String::new()
    } else {
        format!("[tool_calls: {}]", tool_calls.join(", "))
    }
}

/// Visible output of one decoded chat update, logged as one delta line by
/// `--enable-log-outputs`.
#[derive(Debug, Default)]
struct UpdateOutput {
    content: String,
    reasoning: String,
    tool_arguments: String,
}

impl UpdateOutput {
    /// Record one visible text or reasoning block delta.
    fn push_block_delta(&mut self, kind: AssistantBlockKind, delta: &str) {
        match kind {
            AssistantBlockKind::Text => self.content.push_str(delta),
            AssistantBlockKind::Reasoning => self.reasoning.push_str(delta),
            AssistantBlockKind::ToolCall => {
                unreachable!("tool calls must flow through dedicated tool-call events")
            }
        }
    }

    /// Append output released after this update ended.
    fn append(&mut self, other: Self) {
        self.content.push_str(&other.content);
        self.reasoning.push_str(&other.reasoning);
        self.tool_arguments.push_str(&other.tool_arguments);
    }

    /// Describe the update like Python: content, `[reasoning: ...]` and
    /// `[tool_calls: <arguments>]` joined by spaces. `None` when nothing was
    /// visible.
    fn describe(&self) -> Option<String> {
        let mut parts = Vec::new();
        if !self.content.is_empty() {
            parts.push(self.content.clone());
        }
        if !self.reasoning.is_empty() {
            parts.push(format!("[reasoning: {}]", self.reasoning));
        }
        if !self.tool_arguments.is_empty() {
            parts.push(format!("[tool_calls: {}]", self.tool_arguments));
        }
        (!parts.is_empty()).then(|| parts.join(" "))
    }
}

/// Per-update delta logging of one chat stream (`--enable-log-deltas`).
///
/// Each update is held until the next one ends or the stream finishes, so the
/// last delta line carries the finish reason: `vllm-chat` emits `Done` after
/// the last update's `LogprobsDelta`, and a parser may release held-back output
/// in between, which belongs to that last update.
#[derive(Debug)]
pub(crate) struct StreamDeltaLog {
    /// Where the deltas are logged.
    log_outputs: OutputLogConfig,
    /// Output of the update in progress.
    current: UpdateOutput,
    /// The last ended update and its output token IDs.
    ended: Option<(UpdateOutput, Vec<u32>)>,
}

impl StreamDeltaLog {
    pub(crate) fn block_delta(&mut self, kind: AssistantBlockKind, delta: &str) {
        self.current.push_block_delta(kind, delta);
    }
    pub(crate) fn tool_arguments(&mut self, delta: &str) {
        self.current.tool_arguments.push_str(delta);
    }

    pub(crate) fn new(log_outputs: OutputLogConfig) -> Self {
        Self {
            log_outputs,
            current: UpdateOutput::default(),
            ended: None,
        }
    }

    /// End the update in progress, whose output token IDs are `token_ids`,
    /// and log the update before it.
    pub(crate) fn end_update(&mut self, token_ids: &[u32]) {
        if let Some((output, token_ids)) = self.ended.take()
            && let Some(delta) = output.describe()
        {
            self.log_outputs.log(LoggedOutput::StreamDelta, &delta, Some(&token_ids), None);
        }
        self.ended = Some((std::mem::take(&mut self.current), token_ids.to_vec()));
    }

    /// Log the held update, with the stream's finish reason when it finished.
    pub(crate) fn finish(&mut self, finish_reason: Option<&str>) {
        let (mut output, token_ids) = match self.ended.take() {
            Some((output, token_ids)) => (output, Some(token_ids)),
            None => (UpdateOutput::default(), None),
        };
        output.append(std::mem::take(&mut self.current));
        if let Some(delta) = output.describe() {
            self.log_outputs.log(
                LoggedOutput::StreamDelta,
                &delta,
                token_ids.as_deref(),
                finish_reason,
            );
        }
    }
}

impl Drop for StreamDeltaLog {
    /// A stream that fails or is dropped before `Done` still logs the output
    /// already sent to the client.
    fn drop(&mut self) {
        self.finish(None);
    }
}

#[cfg(test)]
mod tests {
    use super::{AssistantBlockKind, UpdateOutput, chat_output_text};
    #[test]
    fn update_output_describes_parts_and_appends_released_output() {
        let mut update = UpdateOutput::default();
        assert_eq!(update.describe(), None);
        update.push_block_delta(AssistantBlockKind::Reasoning, "Need tool.");
        update.push_block_delta(AssistantBlockKind::Text, "Hello");
        update.tool_arguments.push_str("{\"city\": ");
        assert_eq!(
            update.describe().as_deref(),
            Some("Hello [reasoning: Need tool.] [tool_calls: {\"city\": ]")
        );

        // Output a parser releases at finish continues the same parts.
        let mut released = UpdateOutput::default();
        released.push_block_delta(AssistantBlockKind::Text, "<tool_ca");
        update.append(released);
        assert_eq!(
            update.describe().as_deref(),
            Some("Hello<tool_ca [reasoning: Need tool.] [tool_calls: {\"city\": ]")
        );
    }

    #[test]
    fn chat_output_text_prefers_content_then_describes_tool_calls() {
        let tool_calls = [("get_weather", r#"{"city": "Paris"}"#), ("get_time", "{}")];
        assert_eq!(
            chat_output_text("Let me check.", tool_calls),
            "Let me check."
        );
        assert_eq!(
            chat_output_text("", tool_calls),
            r#"[tool_calls: get_weather({"city": "Paris"}), get_time({})]"#
        );
        assert_eq!(chat_output_text("", []), "");
    }
}
