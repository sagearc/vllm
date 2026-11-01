// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod stream;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::json;
use validator::Validate;

use crate::error::{ApiError, chat_submit_error, invalid_request, server_error, text_submit_error};
use crate::routes::openai::chat_completions::{
    ChatCompletionOutput, ChatCompletionRequest, create_chat_completion, lower_chat_request,
};
use crate::routes::openai::utils::types::{
    ChatMessage, ContentPart, Function, FunctionCallResponse, FunctionChoice, MessageContent,
    Normalizable, Tool, ToolCall, ToolChoice, ToolChoiceValue,
};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::resolve_request_context;

#[derive(Debug, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct CountTokensRequest {
    model: String,
    messages: Vec<Message>,
    system: Option<Content>,
    tools: Option<Vec<CustomTool>>,
    tool_choice: Option<MessagesToolChoice>,
}

impl Normalizable for CountTokensRequest {}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
enum Message {
    User { content: Content },
    Assistant { content: Content },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Block {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Map<String, serde_json::Value>,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<Content>,
        #[serde(rename = "is_error")]
        _is_error: Option<bool>,
    },
}

impl Content {
    fn blocks(self) -> Vec<Block> {
        match self {
            Self::Text(text) => vec![Block::Text { text }],
            Self::Blocks(blocks) => blocks,
        }
    }

    fn into_text(self, separator: &str) -> Result<String, ApiError> {
        self.blocks()
            .into_iter()
            .map(|block| match block {
                Block::Text { text } => Ok(text),
                _ => Err(invalid_request!("Only text blocks are supported here")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.join(separator))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CustomTool {
    name: String,
    description: Option<String>,
    input_schema: serde_json::Map<String, serde_json::Value>,
    strict: Option<bool>,
    defer_loading: Option<bool>,
    #[serde(rename = "type")]
    _kind: Option<CustomToolType>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum CustomToolType {
    Custom,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum MessagesToolChoice {
    Auto {
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        disable_parallel_tool_use: Option<bool>,
    },
    None,
    Tool {
        name: String,
        disable_parallel_tool_use: Option<bool>,
    },
}

impl Message {
    fn append(self, messages: &mut Vec<ChatMessage>) -> Result<(), ApiError> {
        let (assistant, content) = match self {
            Self::User { content } => (false, content),
            Self::Assistant { content } => (true, content),
        };
        let mut parts = Vec::new();
        let mut calls = Vec::new();
        for block in content.blocks() {
            match block {
                Block::Text { text } if !text.is_empty() => parts.push(ContentPart::Text { text }),
                Block::Text { .. } => {}
                Block::ToolUse { id, name, input } if assistant => calls.push(ToolCall {
                    id,
                    tool_type: "function".into(),
                    function: FunctionCallResponse {
                        name,
                        arguments: Some(serde_json::Value::Object(input).to_string()),
                    },
                }),
                Block::ToolResult {
                    tool_use_id,
                    content,
                    _is_error: _,
                } if !assistant => {
                    messages.push(ChatMessage::Tool {
                        tool_call_id: tool_use_id,
                        content: MessageContent::Text(
                            content.map(|c| c.into_text("\n")).transpose()?.unwrap_or_default(),
                        ),
                    });
                }
                _ => {
                    return Err(invalid_request!(
                        "tool_use requires an assistant role and tool_result requires a user role"
                    ));
                }
            }
        }
        let content = match parts.as_slice() {
            [] => None,
            [ContentPart::Text { text }] => Some(MessageContent::Text(text.clone())),
            _ => Some(MessageContent::Parts(parts)),
        };
        if assistant {
            messages.push(ChatMessage::Assistant {
                content,
                name: None,
                reasoning: None,
                tool_calls: (!calls.is_empty()).then_some(calls),
            });
        } else if let Some(content) = content {
            messages.push(ChatMessage::User {
                content,
                name: None,
            });
        }
        Ok(())
    }
}

impl CountTokensRequest {
    fn into_chat(self) -> Result<ChatCompletionRequest, ApiError> {
        if self.model.is_empty() || self.messages.is_empty() {
            return Err(invalid_request!("model and messages must be non-empty"));
        }
        let mut messages = Vec::new();
        if let Some(system) = self.system {
            let text = system.into_text("")?;
            if !text.is_empty() {
                messages.push(ChatMessage::System {
                    content: MessageContent::Text(text),
                    name: None,
                });
            }
        }
        for message in self.messages {
            message.append(&mut messages)?;
        }
        let tools: Vec<_> = self
            .tools
            .unwrap_or_default()
            .into_iter()
            .map(|tool| {
                let CustomTool {
                    name,
                    description,
                    mut input_schema,
                    strict,
                    defer_loading,
                    _kind: _,
                } = tool;
                input_schema.entry("type".to_owned()).or_insert(json!("object"));
                Tool {
                    tool_type: "function".into(),
                    defer_loading: None,
                    function: Function {
                        name,
                        description,
                        parameters: input_schema.into(),
                        strict,
                        defer_loading,
                    },
                }
            })
            .collect();
        let (tool_choice, parallel) = match self.tool_choice {
            None => (None, None),
            Some(MessagesToolChoice::Auto {
                disable_parallel_tool_use,
            }) => (
                Some(ToolChoice::Value(ToolChoiceValue::Auto)),
                disable_parallel_tool_use,
            ),
            Some(MessagesToolChoice::Any {
                disable_parallel_tool_use,
            }) => (
                Some(ToolChoice::Value(ToolChoiceValue::Required)),
                disable_parallel_tool_use,
            ),
            Some(MessagesToolChoice::None) => {
                (Some(ToolChoice::Value(ToolChoiceValue::None)), None)
            }
            Some(MessagesToolChoice::Tool {
                name,
                disable_parallel_tool_use,
            }) => (
                Some(ToolChoice::Function {
                    tool_type: "function".into(),
                    function: FunctionChoice { name },
                }),
                disable_parallel_tool_use,
            ),
        };
        let mut request = ChatCompletionRequest {
            model: Some(self.model),
            messages,
            tools: (!tools.is_empty()).then_some(tools),
            tool_choice,
            parallel_tool_calls: parallel.map(|disabled| !disabled),
            ..Default::default()
        };
        request.normalize();
        request.validate().map_err(|error| invalid_request!("{error}"))?;
        Ok(request)
    }
}

pub struct AnthropicError(ApiError);

impl From<ApiError> for AnthropicError {
    fn from(error: ApiError) -> Self {
        Self(error)
    }
}

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        (self.0.status_code(), Json(error_body(&self.0))).into_response()
    }
}

fn error_body(error: &ApiError) -> serde_json::Value {
    let kind = match error.status_code().as_u16() {
        400 => "invalid_request_error",
        404 => "not_found_error",
        409 => "conflict_error",
        503 => "overloaded_error",
        _ => "api_error",
    };
    json!({"type": "error", "error": {"type": kind,
        "message": error.to_error_response().error.message}})
}

pub async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<ValidatedJson<CountTokensRequest>, ApiError>,
) -> Result<Json<serde_json::Value>, AnthropicError> {
    let request = body?.0.into_chat()?;
    let model = state.resolve_model_with_loras(request.model.as_deref()).await;
    let request = lower_chat_request(request, &model, resolve_request_context(&headers, None))?;
    let text = state
        .chat
        .request_processor()
        .prepare_for_tokenization(request)
        .await
        .map_err(|error| chat_submit_error("count_tokens failed", error))?;
    let tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(text)
        .map_err(|error| text_submit_error("count_tokens failed", error))?;
    Ok(Json(json!({"input_tokens": tokens.len(),
        "context_management": {"original_input_tokens": tokens.len()}})))
}

#[derive(Debug, Deserialize, Validate)]
pub struct MessagesRequest {
    #[serde(flatten)]
    input: CountTokensRequest,
    #[validate(range(min = 1))]
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    stream: Option<bool>,
}

impl Normalizable for MessagesRequest {}

impl MessagesRequest {
    fn into_chat(self) -> Result<ChatCompletionRequest, ApiError> {
        let mut request = self.input.into_chat()?;
        request.stream = self.stream.unwrap_or(false);
        if request.stream {
            request.stream_options = Some(crate::routes::openai::utils::types::StreamOptions {
                include_usage: Some(true),
                continuous_usage_stats: Some(true),
            });
        }
        request.max_completion_tokens = Some(self.max_tokens);
        request.temperature = self.temperature;
        request.top_p = self.top_p;
        request.top_k = self.top_k;
        request.stop = self
            .stop_sequences
            .map(crate::routes::openai::utils::types::StringOrArray::Array);
        request.validate().map_err(|error| invalid_request!("{error}"))?;
        Ok(request)
    }
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<ValidatedJson<MessagesRequest>, ApiError>,
) -> Result<Response, AnthropicError> {
    let request = body?.0.into_chat()?;
    let keep_alive = state.api_server_options.sse_keep_alive_interval;
    let response = match create_chat_completion(state, headers, request).await? {
        ChatCompletionOutput::Full(response) => response,
        ChatCompletionOutput::Stream(stream) => {
            return Ok(crate::utils::sse_response(
                stream::messages_stream(stream),
                keep_alive,
            ));
        }
    };
    let choice = &response.choices[0];
    let (reason, sequence) = stop_reason(
        choice.finish_reason.as_deref(),
        choice.stop_reason.as_ref(),
        !choice.message.tool_calls.is_empty(),
        true,
    )?;
    if choice.message.reasoning.is_some() {
        return Err(invalid_request!("Reasoning output is not supported yet").into());
    }
    let mut content = Vec::new();
    if let Some(text) = &choice.message.content {
        content.push(json!({"type": "text", "text": text}));
    }
    for call in &choice.message.tool_calls {
        content.push(
            json!({"type": "tool_use", "id": call.id, "name": call.function.name,
            "input": tool_input(call.function.arguments.as_deref().unwrap_or(""))?}),
        );
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }
    let mut message = json!({
        "id": response.id, "type": "message", "role": "assistant", "model": response.model,
        "content": content,
        "stop_reason": reason, "stop_sequence": sequence,
        "usage": usage(response.usage.as_ref()),
    });
    if let Some(params) = response.kv_transfer_params {
        message["kv_transfer_params"] = params;
    }
    if let Some(params) = response.ec_transfer_params {
        message["ec_transfer_params"] = params;
    }
    Ok(Json(message).into_response())
}

fn stop_reason<'a>(
    finish: Option<&str>,
    stop: Option<&'a serde_json::Value>,
    tool_used: bool,
    tool_complete: bool,
) -> Result<(&'static str, Option<&'a str>), ApiError> {
    match finish {
        Some("length") => Ok(("max_tokens", None)),
        Some("stop" | "tool_calls") if tool_used && tool_complete => Ok(("tool_use", None)),
        Some("stop" | "tool_calls") => match stop.and_then(serde_json::Value::as_str) {
            Some(text) => Ok(("stop_sequence", Some(text))),
            None if tool_used => Ok(("tool_use", None)),
            None => Ok(("end_turn", None)),
        },
        _ => Err(server_error!("Generation did not complete successfully")),
    }
}

fn usage(usage: Option<&crate::routes::openai::utils::types::Usage>) -> serde_json::Value {
    let Some(usage) = usage else {
        return json!({"input_tokens": 0, "output_tokens": 0});
    };
    let mut result = json!({"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens.unwrap_or(0)});
    if let Some(details) = &usage.prompt_tokens_details {
        let cached = details.cached_tokens;
        let created = details.created_cache_tokens;
        result["input_tokens"] =
            json!(usage.prompt_tokens.saturating_sub(cached).saturating_sub(created));
        result["cache_read_input_tokens"] = json!(cached);
        result["cache_creation_input_tokens"] = json!(created);
    }
    result
}

fn tool_input(arguments: &str) -> Result<serde_json::Value, ApiError> {
    if arguments.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(arguments)
        .map(serde_json::Value::Object)
        .map_err(|_| server_error!("Generated tool arguments are not a complete JSON object"))
}
