// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use serde::Serialize;
use thiserror_ext::AsReport as _;
use vllm_engine_core_client::protocol::multimodal::MmFeatures;
use vllm_text::TextRequest;

use crate::DEFAULT_REQUEST_BODY_LIMIT_BYTES;
use crate::error::{ApiError, text_submit_error};
use crate::lora::LoraModelResolution;
use crate::render::RenderState;
use crate::routes::inference::generate::{
    GenerateRequest, GenerateSamplingParams, validate_request_compat as validate_generate_request,
};
use crate::routes::openai::utils::types::{ListModelsResponse, ModelObject, StreamOptions};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::openai::{
    ChatCompletionRequest, CompletionRequest, lower_chat_request, lower_completion_request,
};
use crate::utils::{resolve_request_context, unix_timestamp};

pub(crate) fn build_router(state: Arc<RenderState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ping", get(health).post(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions/render", post(render_chat))
        .route("/v1/completions/render", post(render_completion))
        .with_state(state)
        .layer(DefaultBodyLimit::max(DEFAULT_REQUEST_BODY_LIMIT_BYTES))
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn list_models(State(state): State<Arc<RenderState>>) -> Json<ListModelsResponse> {
    let created = unix_timestamp() as i64;
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: state
            .served_model_names
            .iter()
            .map(|model| ModelObject {
                id: model.clone(),
                object: "model".to_string(),
                created,
                owned_by: "vllm".to_string(),
                root: Some(state.model.clone()),
                parent: None,
                max_model_len: Some(state.text.max_model_len()),
            })
            .collect(),
    })
}

fn model_resolution(state: &RenderState) -> LoraModelResolution {
    LoraModelResolution {
        model_names: state.served_model_names.clone(),
        lora_request: None,
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct RenderResponse {
    #[serde(flatten)]
    request: GenerateRequest,
    features: Option<RenderMultimodalFeatures>,
}

#[derive(Debug, Serialize)]
struct RenderMultimodalFeatures {
    mm_hashes: BTreeMap<String, Vec<String>>,
    mm_placeholders: BTreeMap<String, Vec<RenderPlaceholderRange>>,
}

#[derive(Debug, Serialize)]
struct RenderPlaceholderRange {
    offset: usize,
    length: usize,
}

fn render_multimodal_features(
    features: MmFeatures,
) -> Result<Option<RenderMultimodalFeatures>, ApiError> {
    if features.is_empty() {
        return Ok(None);
    }

    let mut mm_hashes = BTreeMap::<String, Vec<String>>::new();
    let mut mm_placeholders = BTreeMap::<String, Vec<RenderPlaceholderRange>>::new();
    for feature in features {
        let hash = feature.mm_hash.ok_or_else(|| {
            ApiError::server_error("rendered multimodal feature is missing its raw-media hash")
        })?;
        mm_hashes.entry(feature.modality.clone()).or_default().push(hash);
        mm_placeholders
            .entry(feature.modality)
            .or_default()
            .push(RenderPlaceholderRange {
                offset: feature.mm_position.offset,
                length: feature.mm_position.length,
            });
    }

    Ok(Some(RenderMultimodalFeatures {
        mm_hashes,
        mm_placeholders,
    }))
}

fn lower_render_request(
    state: &RenderState,
    text_request: TextRequest,
    model: String,
    stream: bool,
    stream_options: Option<StreamOptions>,
) -> Result<RenderResponse, ApiError> {
    let prepared = state
        .text
        .prepare(text_request)
        .map_err(|error| text_submit_error("failed to prepare render request", error))?;
    let token_ids = prepared.generate_request.prompt_token_ids;
    let features =
        render_multimodal_features(prepared.generate_request.mm_features.unwrap_or_default())?;
    let text_request = prepared.text_request;

    let request = GenerateRequest {
        request_id: Some(text_request.request_id),
        model: Some(model),
        token_ids,
        sampling_params: GenerateSamplingParams {
            n: None,
            inner: text_request.sampling_params,
        },
        stream,
        stream_options,
        cache_salt: text_request.cache_salt,
        priority: text_request.priority,
        kv_transfer_params: None,
        ec_transfer_params: None,
        content_parts: None,
        other: Default::default(),
    };
    validate_generate_request(&request, &state.served_model_names)?;
    Ok(RenderResponse { request, features })
}

async fn render_chat(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Result<Json<RenderResponse>, ApiError> {
    let model = body.model.clone();
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let chat_request = lower_chat_request(body, &model_resolution(&state), request_context)?;
    let (text_request, _) = state
        .chat
        .prepare(chat_request)
        .await
        .map_err(|error| ApiError::invalid_request(error.to_report_string(), None))?;
    Ok(Json(lower_render_request(
        &state,
        text_request,
        model,
        stream,
        stream_options,
    )?))
}

async fn render_completion(
    State(state): State<Arc<RenderState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Result<Json<Vec<RenderResponse>>, ApiError> {
    let model = body.model.clone();
    let stream = body.stream;
    let stream_options = body.stream_options.clone();
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let tokenizer = state.text.tokenizer();
    let text_request = lower_completion_request(
        body,
        &model_resolution(&state),
        request_context,
        tokenizer.as_ref(),
    )?;
    Ok(Json(vec![lower_render_request(
        &state,
        text_request,
        model,
        stream,
        stream_options,
    )?]))
}
