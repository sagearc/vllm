// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use asynk_strim_attr::{TryYielder, try_stream};

use futures::StreamExt as _;
use serde::Serialize;
use vllm_chat::{AssistantBlockKind, AssistantMessageExt as _, ChatEvent, ChatEventStream};
use vllm_llm::TokenUsage;

use super::{MessagesResponse, MessagesUsage, ResponseBlock, stop_reason, usage};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, chat_submit_error, server_error};
use crate::routes::output_log::{LoggedOutput, StreamDeltaLog};

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum MessagesEvent {
    MessageStart {
        message: Box<MessagesResponse>,
    },
    ContentBlockStart {
        index: usize,
        content_block: ResponseBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: MessagesDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: StopDelta,
        usage: MessagesUsage,
        #[serde(skip_serializing_if = "Option::is_none")]
        kv_transfer_params: Option<Box<serde_json::Value>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ec_transfer_params: Option<Box<serde_json::Value>>,
    },
    MessageStop,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum MessagesDelta {
    TextDelta { text: String },
}

#[derive(Serialize)]
pub(crate) struct StopDelta {
    stop_reason: &'static str,
    stop_sequence: Option<String>,
}

#[try_stream]
pub(super) async fn messages_stream(
    mut stream: ChatEventStream,
    request_id: String,
    model: String,
    options: ApiServerOptions,
    mut y: TryYielder<MessagesEvent, ApiError>,
) -> Result<(), ApiError> {
    let mut delta_log = options
        .log_outputs
        .logs_deltas()
        .then(|| StreamDeltaLog::new(options.log_outputs));
    let mut saw_block = false;
    while let Some(next) = stream.next().await {
        let next = next.map_err(|error| chat_submit_error("Messages stream failed", error))?;
        match next {
            ChatEvent::Start {
                prompt_token_ids, ..
            } => {
                y.yield_ok(MessagesEvent::MessageStart {
                    message: Box::new(MessagesResponse {
                        id: request_id.clone(),
                        message_type: "message",
                        role: "assistant",
                        model: model.clone(),
                        content: vec![],
                        stop_reason: None,
                        stop_sequence: None,
                        usage: usage(
                            TokenUsage {
                                prompt_token_count: prompt_token_ids.len(),
                                ..Default::default()
                            },
                            false,
                        ),
                        kv_transfer_params: None,
                        ec_transfer_params: None,
                    }),
                })
                .await;
            }
            ChatEvent::BlockStart {
                index,
                kind: AssistantBlockKind::Text,
            } => {
                saw_block = true;
                y.yield_ok(MessagesEvent::ContentBlockStart {
                    index,
                    content_block: ResponseBlock::Text {
                        text: String::new(),
                    },
                })
                .await;
            }
            ChatEvent::BlockDelta {
                index,
                kind: AssistantBlockKind::Text,
                delta,
                ..
            } => {
                if let Some(log) = &mut delta_log {
                    log.block_delta(AssistantBlockKind::Text, &delta);
                }
                y.yield_ok(MessagesEvent::ContentBlockDelta {
                    index,
                    delta: MessagesDelta::TextDelta { text: delta },
                })
                .await;
            }
            ChatEvent::BlockEnd { index, .. } => {
                y.yield_ok(MessagesEvent::ContentBlockStop { index }).await;
            }
            ChatEvent::LogprobsDelta { token_ids, .. } => {
                if let Some(log) = &mut delta_log {
                    log.end_update(&token_ids);
                }
            }
            ChatEvent::Done {
                message,
                usage: final_usage,
                finish_reason,
                kv_transfer_params,
                ec_transfer_params,
                ..
            } => {
                let (reason, sequence) = stop_reason(&finish_reason)?;
                if let Some(log) = &mut delta_log {
                    log.finish(Some(reason));
                }
                if options.enable_log_requests {
                    tracing::info!(
                        model,
                        prompt_tokens = final_usage.prompt_token_count,
                        output_tokens = final_usage.output_token_count,
                        finish_reason = reason,
                        "Messages finished"
                    );
                }
                if options.log_outputs.logs_complete() {
                    options.log_outputs.log(
                        LoggedOutput::StreamComplete,
                        &message.text(),
                        None,
                        Some(reason),
                    );
                }
                if !saw_block {
                    y.yield_ok(MessagesEvent::ContentBlockStart {
                        index: 0,
                        content_block: ResponseBlock::Text {
                            text: String::new(),
                        },
                    })
                    .await;
                    y.yield_ok(MessagesEvent::ContentBlockStop { index: 0 }).await;
                }
                y.yield_ok(MessagesEvent::MessageDelta {
                    delta: StopDelta {
                        stop_reason: reason,
                        stop_sequence: sequence.map(str::to_owned),
                    },
                    usage: usage(final_usage.engine, options.enable_prompt_tokens_details),
                    kv_transfer_params,
                    ec_transfer_params,
                })
                .await;
                y.yield_ok(MessagesEvent::MessageStop).await;
                return Ok(());
            }
            _ => {
                return Err(server_error!(
                    "Reasoning and tool output are not supported yet"
                ));
            }
        }
    }
    Err(server_error!(
        "Messages stream ended without a terminal result"
    ))
}
