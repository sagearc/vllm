// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::convert::Infallible;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::response::sse::Event;
use futures::{Stream, StreamExt as _, pin_mut};
use serde_json::{Value, json};

use super::{error_body, stop_reason, tool_input, usage};
use crate::error::{ApiError, server_error};
use crate::routes::openai::chat_completions::types::ChatCompletionStreamResponse;
use crate::routes::openai::utils::types::ToolCallDelta;

fn event(body: Value) -> Event {
    Event::default()
        .event(body["type"].as_str().expect("event type"))
        .json_data(body)
        .expect("JSON event")
}

#[try_stream]
pub(super) async fn messages_stream(
    stream: impl Stream<Item = Result<ChatCompletionStreamResponse, ApiError>>,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    pin_mut!(stream);
    let mut started = false;
    let mut blocks = Blocks::default();
    let mut final_usage = usage(None);
    let mut finish = None;
    let mut matched_stop = None;
    let mut transfer = None;
    let mut failure = None;
    while let Some(next) = stream.next().await {
        let chunk = match next {
            Ok(chunk) => chunk,
            Err(error) => {
                failure = Some(error);
                break;
            }
        };
        if chunk.usage.is_some() {
            final_usage = usage(chunk.usage.as_ref());
        }
        if !started {
            y.yield_ok(event(json!({"type": "message_start", "message": {
                "id": chunk.envelope.id(), "type": "message", "role": "assistant",
                "model": chunk.envelope.model(), "content": [], "stop_reason": null,
                "stop_sequence": null, "usage": final_usage,
            }})))
            .await;
            started = true;
        }
        transfer = chunk.kv_transfer_params.or(transfer);
        for choice in chunk.choices {
            if choice.delta.reasoning.is_some() {
                failure = Some(server_error!("Reasoning output is not supported yet"));
                break;
            }
            let mut events = Vec::new();
            if let Some(text) = choice.delta.content.filter(|text| !text.is_empty()) {
                blocks.text(text, &mut events);
            }
            for call in choice.delta.tool_calls.unwrap_or_default() {
                if let Err(error) = blocks.tool(call, &mut events) {
                    failure = Some(error);
                    break;
                }
            }
            for body in events {
                y.yield_ok(event(body)).await;
            }
            if choice.finish_reason.is_some() {
                finish = choice.finish_reason;
                matched_stop = choice.stop_reason;
            }
        }
        if failure.is_some() {
            break;
        }
    }
    let reason = stop_reason(
        finish.as_deref(),
        matched_stop.as_ref(),
        blocks.tool_used,
        tool_input(&blocks.tool_arguments).is_ok(),
    );
    let mut events = Vec::new();
    blocks.finish(&mut events);
    if failure.is_none() && reason.is_ok() && blocks.index == 0 {
        blocks.text(String::new(), &mut events);
        blocks.close(&mut events);
    }
    for body in events {
        y.yield_ok(event(body)).await;
    }
    match failure.map_or(reason, Err) {
        Err(error) => y.yield_ok(event(error_body(&error))).await,
        Ok((reason, sequence)) => {
            let mut delta = json!({"type": "message_delta", "delta": {
                "stop_reason": reason, "stop_sequence": sequence}, "usage": final_usage});
            if let Some(transfer) = transfer {
                delta["kv_transfer_params"] = transfer;
            }
            y.yield_ok(event(delta)).await;
            y.yield_ok(event(json!({"type": "message_stop"}))).await;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Text,
    Tool(u32),
}

#[derive(Default)]
struct Blocks {
    index: usize,
    open: Option<Kind>,
    tool_arguments: String,
    tool_used: bool,
    pending_text: String,
}

impl Blocks {
    fn close(&mut self, events: &mut Vec<Value>) {
        if self.open.take().is_some() {
            events.push(json!({"type": "content_block_stop", "index": self.index}));
            self.index += 1;
        }
    }

    fn text(&mut self, text: String, events: &mut Vec<Value>) {
        if matches!(self.open, Some(Kind::Tool(_))) {
            self.pending_text.push_str(&text);
            return;
        }
        if self.open != Some(Kind::Text) {
            self.close(events);
            events.push(json!({"type": "content_block_start", "index": self.index,
                "content_block": {"type": "text", "text": ""}}));
            self.open = Some(Kind::Text);
        }
        if !text.is_empty() {
            events.push(json!({"type": "content_block_delta", "index": self.index,
                "delta": {"type": "text_delta", "text": text}}));
        }
    }

    fn finish(&mut self, events: &mut Vec<Value>) {
        self.close(events);
        if !self.pending_text.is_empty() {
            let text = std::mem::take(&mut self.pending_text);
            self.text(text, events);
            self.close(events);
        }
    }

    fn tool(&mut self, call: ToolCallDelta, events: &mut Vec<Value>) -> Result<(), ApiError> {
        let Some(function) = call.function else {
            return Ok(());
        };
        if let Some(id) = call.id {
            self.finish(events);
            let name = function.name.ok_or_else(|| server_error!("Tool call has no name"))?;
            events.push(json!({"type": "content_block_start", "index": self.index,
                "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}}));
            self.open = Some(Kind::Tool(call.index));
            self.tool_used = true;
            self.tool_arguments.clear();
        }
        if let Some(arguments) = function.arguments.filter(|args| !args.is_empty()) {
            if self.open != Some(Kind::Tool(call.index)) {
                return Err(server_error!(
                    "Tool arguments arrived without their tool call"
                ));
            }
            self.tool_arguments.push_str(&arguments);
            events.push(json!({"type": "content_block_delta", "index": self.index,
                "delta": {"type": "input_json_delta", "partial_json": arguments}}));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routes::openai::utils::types::FunctionCallDelta;

    #[test]
    fn text_between_argument_deltas_does_not_truncate_a_tool_call() {
        let mut blocks = Blocks::default();
        let mut events = Vec::new();
        for (id, name, arguments, text) in [
            (
                Some("call1"),
                Some("weather"),
                "{\"city\":\"Paris\"",
                Some("after tool"),
            ),
            (None, None, "}", None),
        ] {
            blocks
                .tool(
                    ToolCallDelta {
                        index: 0,
                        id: id.map(str::to_owned),
                        tool_type: None,
                        function: Some(FunctionCallDelta {
                            name: name.map(str::to_owned),
                            arguments: Some(arguments.into()),
                        }),
                    },
                    &mut events,
                )
                .unwrap();
            if let Some(text) = text {
                blocks.text(text.into(), &mut events);
            }
        }
        blocks.finish(&mut events);
        assert_eq!(events[2]["delta"]["partial_json"], "}");
        assert_eq!(events[3], json!({"type": "content_block_stop", "index": 0}));
        assert_eq!(events[5]["delta"]["text"], "after tool");
        assert_eq!(
            tool_input(&blocks.tool_arguments).unwrap(),
            json!({"city": "Paris"})
        );
    }
}
