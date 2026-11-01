// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::convert::Infallible;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::response::sse::Event;
use futures::{Stream, StreamExt as _, pin_mut};
use serde_json::{Value, json};

use super::{error_body, stop_reason, usage};
use crate::error::{ApiError, server_error};
use crate::routes::openai::chat_completions::types::ChatCompletionStreamResponse;

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
    let mut block_open = false;
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
            if choice.delta.reasoning.is_some() || choice.delta.tool_calls.is_some() {
                failure = Some(server_error!(
                    "Reasoning and tool output are not supported yet"
                ));
                break;
            }
            if let Some(text) = choice.delta.content.filter(|text| !text.is_empty()) {
                if !block_open {
                    y.yield_ok(event(json!({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}})))
                        .await;
                    block_open = true;
                }
                y.yield_ok(event(json!({"type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": text}})))
                    .await;
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
    let reason = stop_reason(finish.as_deref(), matched_stop.as_ref());
    if failure.is_none() && reason.is_ok() && !block_open {
        y.yield_ok(event(json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}})))
            .await;
        block_open = true;
    }
    if block_open {
        y.yield_ok(event(json!({"type": "content_block_stop", "index": 0}))).await;
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
