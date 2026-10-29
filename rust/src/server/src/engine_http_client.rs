// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

use asynk_strim_attr::{TryYielder, try_stream};
use eventsource_stream::Eventsource as _;
use futures::{StreamExt as _, future::BoxFuture};
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror_ext::AsReport as _;
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::sampling_mask::SamplingMask;
use vllm_llm::{
    Error, FinishReason, GenerateOutput, GeneratePromptInfo, Logprobs, PositionLogprobs,
    RequestTimestamps, Result, TokenLogprob,
};
use vllm_text::{EngineClient, GenerationRequest, TextRequestProcessor, TokenStream};

use crate::routes::inference::generate::types::{
    GenerateLogProbs, GenerateOutputMode, GenerateRequest as HttpGenerateRequest,
    GenerateResponseStreamChoice, GenerateSamplingParams,
};
use crate::routes::openai::utils::types::{PromptTokenUsageInfo, StreamOptions};

#[cfg(test)]
mod tests;

#[derive(Deserialize)]
#[serde(untagged)]
enum GenerateEvent {
    Error { error: Value },
    Tokens(TokenResponse),
}

/// Read token outputs and usage from the generate API's SSE chunks.
#[derive(Deserialize)]
struct TokenResponse {
    choices: Vec<GenerateResponseStreamChoice>,
    usage: Option<ResponseUsage>,
    kv_transfer_params: Option<Value>,
    ec_transfer_params: Option<Value>,
}

#[derive(Deserialize)]
struct ResponseUsage {
    prompt_tokens_details: Option<PromptTokenUsageInfo>,
}

type ActiveRequests = Arc<Mutex<HashMap<Uuid, (String, CancellationToken)>>>;

/// HTTP token generation through a gateway that selects the engine per request.
pub(crate) struct EngineHTTPClient {
    client: reqwest::Client,
    url: Url,
    model: String,
    max_model_len: u32,
    stream_interval: NonZeroU32,
    active: ActiveRequests,
}

pub(crate) fn generate_url(base: &str) -> anyhow::Result<Url> {
    let mut url = Url::parse(base)?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https") && url.host_str().is_some(),
        "generation URL must use HTTP or HTTPS"
    );
    anyhow::ensure!(
        url.query().is_none() && url.fragment().is_none(),
        "generation URL must not contain a query or fragment"
    );
    let path = format!("{}/inference/v1/generate", url.path().trim_end_matches('/'));
    url.set_path(&path);
    Ok(url)
}

impl EngineHTTPClient {
    pub(crate) fn new(
        base: &str,
        model: String,
        stream_interval: NonZeroU32,
        max_model_len: u32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            // Redirects and retries can submit generation to a second engine.
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never())
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()?,
            url: generate_url(base)?,
            model,
            max_model_len,
            stream_interval,
            active: Default::default(),
        })
    }
}

struct RequestGuard {
    key: Uuid,
    active: ActiveRequests,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.active.lock().expect("active requests mutex").remove(&self.key);
    }
}

impl EngineClient for EngineHTTPClient {
    fn generate<'a>(
        &'a self,
        request: GenerationRequest,
        _processor: &'a TextRequestProcessor,
    ) -> BoxFuture<'a, vllm_text::Result<TokenStream>> {
        Box::pin(async move {
            let body = request_body(&self.model, self.stream_interval, &request)?;
            let prepared = request;
            let mut http = self
                .client
                .post(self.url.clone())
                .header("X-Request-Id", &prepared.request_id)
                .json(&body);
            if let Some(rank) = prepared.data_parallel_rank {
                http = http.header("X-data-parallel-rank", rank);
            }
            if let Some(session_id) = &prepared.session_id {
                http = http.header("X-Session-ID", session_id);
            }

            let cancel = CancellationToken::new();
            let key = Uuid::new_v4();
            self.active
                .lock()
                .expect("active requests mutex")
                .insert(key, (prepared.request_id.clone(), cancel.clone()));
            let guard = RequestGuard {
                key,
                active: self.active.clone(),
            };
            let response = tokio::select! {
                _ = cancel.cancelled() => return Err(remote_error(499, "generation cancelled").into()),
                response = http.send() => response.map_err(transport_error)?,
            };
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let message = tokio::select! {
                    _ = cancel.cancelled() => "generation cancelled".to_owned(),
                    body = response.text() => body.map_err(transport_error)?,
                };
                return Err(remote_error(status, message).into());
            }
            Ok(receive(response, prepared, cancel, guard).boxed())
        })
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }
    fn model_dtype(&self) -> Option<ModelDtype> {
        None
    }
    fn is_healthy(&self) -> bool {
        true
    }
    fn supports_lora(&self) -> bool {
        false
    }

    fn abort<'a>(&'a self, ids: &'a [String]) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            for (id, cancel) in self.active.lock().expect("active requests mutex").values() {
                if ids.is_empty() || ids.contains(id) {
                    cancel.cancel();
                }
            }
            Ok(())
        })
    }

    fn shutdown(self: Box<Self>) -> BoxFuture<'static, Result<()>> {
        Box::pin(async move { self.abort(&[]).await })
    }
}

#[derive(Debug, thiserror::Error)]
#[error("HTTP generation returned {status}: {message}")]
pub(crate) struct HttpGenerationError {
    pub(crate) status: u16,
    pub(crate) message: String,
}

fn remote_error(status: u16, message: impl Into<String>) -> Error {
    Error::Backend {
        source: Box::new(HttpGenerationError {
            status,
            message: message.into(),
        }),
    }
}

fn transport_error(error: reqwest::Error) -> Error {
    remote_error(502, error.to_report_string())
}

fn request_body(
    model: &str,
    stream_interval: NonZeroU32,
    prepared: &GenerationRequest,
) -> Result<Value> {
    let sampling = &prepared.sampling_params;
    // TODO: support prompt logprobs in the HTTP backend.
    if sampling.prompt_logprobs.is_some() {
        return Err(remote_error(
            400,
            "HTTP generation does not support prompt_logprobs",
        ));
    }
    if prepared.mm_features.is_some()
        || prepared.lora_request.is_some()
        || prepared.kv_hints.is_some()
        || sampling.bad_words_token_ids.is_some()
        || sampling.prompt_logprob_token_ids.is_some()
        || sampling.logprob_token_ids.as_ref().is_some_and(|ids| !ids.is_empty())
    {
        return Err(remote_error(
            400,
            "HTTP generation does not support multimodal features, LoRA, KV hints, pretokenized bad words, or selective logprobs (logprob_token_ids / prompt_logprob_token_ids)",
        ));
    }
    let mut sampling = sampling.clone();
    sampling.stream_interval =
        Some(sampling.stream_interval.unwrap_or(NonZeroU32::MIN).max(stream_interval));
    let mut extra = sampling.vllm_xargs.take().unwrap_or_default();
    let kv_transfer_params = extra
        .remove("kv_transfer_params")
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| remote_error(400, e.to_report_string()))?;
    let ec_transfer_params = extra
        .remove("ec_transfer_params")
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| remote_error(400, e.to_report_string()))?;
    let wire = HttpGenerateRequest {
        request_id: Some(prepared.request_id.clone()),
        model: Some(model.to_owned()),
        token_ids: prepared.prompt_token_ids.clone(),
        sampling_params: GenerateSamplingParams {
            n: Some(1),
            stop: None,
            include_stop_str_in_output: None,
            trace_decode_token_ids: None,
            inner: sampling,
        },
        stream: true,
        stream_options: Some(StreamOptions {
            include_usage: Some(true),
            continuous_usage_stats: Some(true),
        }),
        cache_salt: prepared.cache_salt.clone(),
        priority: prepared.priority,
        kv_transfer_params,
        ec_transfer_params,
        content_parts: None,
        return_token_ids: None,
        output_mode: Some(GenerateOutputMode::Tokens),
        reasoning_ended: prepared.reasoning_ended,
        reasoning_parser_kwargs: prepared.reasoning_parser_kwargs.clone(),
        other: Default::default(),
    };
    let mut body =
        serde_json::to_value(wire).map_err(|e| remote_error(400, e.to_report_string()))?;
    // The frontend owns detokenization and stop-string matching.
    body["sampling_params"]["detokenize"] = json!(false);
    if !extra.is_empty() {
        body["sampling_params"]["extra_args"] = json!(extra);
    }
    Ok(body)
}

#[try_stream]
async fn receive(
    response: reqwest::Response,
    request: GenerationRequest,
    cancel: CancellationToken,
    guard: RequestGuard,
    mut y: TryYielder<GenerateOutput, Error>,
) -> Result<()> {
    let _guard = guard;
    let mut prompt = Some(Box::new(GeneratePromptInfo {
        prompt_token_ids: request.prompt_token_ids.into(),
        prompt_logprobs: None,
        prompt_token_id_logprobs: None,
    }));
    let mut cached_tokens = 0;
    let mut events = response.bytes_stream().eventsource();
    let mut terminal = None;
    loop {
        let event = tokio::select! {
            _ = cancel.cancelled() => {
                y.yield_ok(aborted(&request.request_id, prompt.take())).await;
                return Ok(());
            }
            event = events.next() => event,
        };
        let event = event
            .ok_or_else(|| remote_error(502, "generate stream closed before [DONE]"))?
            .map_err(|e| remote_error(502, e.to_report_string()))?;
        if event.data == "[DONE]" {
            let mut output: GenerateOutput = terminal.ok_or_else(|| {
                remote_error(502, "generate stream ended without a finish reason")
            })?;
            output.cached_token_count = cached_tokens;
            y.yield_ok(output).await;
            return Ok(());
        }
        let event = serde_json::from_str::<GenerateEvent>(&event.data)
            .map_err(|e| remote_error(502, e.to_report_string()))?;
        let body = match event {
            GenerateEvent::Error { error } => return Err(remote_error(502, error.to_string())),
            GenerateEvent::Tokens(body) => body,
        };
        if let Some(output) =
            convert_response(body, &request.request_id, &mut prompt, &mut cached_tokens)?
        {
            if terminal.is_some() {
                return Err(remote_error(502, "tokens received after terminal output"));
            }
            if output.finished() {
                terminal = Some(output);
            } else {
                y.yield_ok(output).await;
            }
        }
    }
}

fn convert_response(
    body: TokenResponse,
    request_id: &str,
    prompt: &mut Option<Box<GeneratePromptInfo>>,
    cached_tokens: &mut usize,
) -> Result<Option<GenerateOutput>> {
    if let Some(details) = body.usage.and_then(|u| u.prompt_tokens_details) {
        *cached_tokens = (*cached_tokens).max(details.cached_tokens);
    }
    if body.choices.is_empty() {
        return Ok(None);
    }
    if body.choices.len() != 1 || body.choices[0].index != 0 {
        return Err(remote_error(
            502,
            "expected one generate choice with index 0",
        ));
    }
    let choice = body.choices.into_iter().next().expect("one choice");
    let finish_reason = match choice.finish_reason.as_deref() {
        None => None,
        Some("stop") => Some(FinishReason::Stop(choice.stop_reason)),
        Some("length") => Some(FinishReason::Length),
        Some("abort") => Some(FinishReason::Abort),
        Some("repetition") => Some(FinishReason::Repetition(choice.stop_reason)),
        Some(reason) => {
            return Err(remote_error(
                502,
                format!("generate finished with {reason}"),
            ));
        }
    };
    let logprobs = choice.logprobs.map(output_logprobs);
    if logprobs.as_ref().is_some_and(|p| p.positions.len() != choice.token_ids.len()) {
        return Err(remote_error(
            502,
            "generate logprobs do not align with token IDs",
        ));
    }
    if choice
        .sampling_mask
        .as_ref()
        .is_some_and(|rows| rows.len() != choice.token_ids.len())
    {
        return Err(remote_error(
            502,
            "generate sampling masks do not align with token IDs",
        ));
    }
    Ok(Some(GenerateOutput {
        request_id: request_id.to_owned(),
        prompt_info: prompt.take(),
        token_ids: choice.token_ids,
        logprobs,
        finish_reason,
        cached_token_count: *cached_tokens,
        kv_transfer_params: body.kv_transfer_params.map(Box::new),
        ec_transfer_params: body.ec_transfer_params.map(Box::new),
        sampling_mask: choice.sampling_mask.map(|rows| SamplingMask { rows }),
        spec_decode_metrics: None,
        weight_versions: choice.weight_versions,
        routed_experts: None,
        timestamps: RequestTimestamps::default(),
    }))
}

fn aborted(request_id: &str, prompt_info: Option<Box<GeneratePromptInfo>>) -> GenerateOutput {
    GenerateOutput {
        request_id: request_id.to_owned(),
        prompt_info,
        token_ids: Vec::new(),
        logprobs: None,
        finish_reason: Some(FinishReason::Abort),
        cached_token_count: 0,
        kv_transfer_params: None,
        ec_transfer_params: None,
        sampling_mask: None,
        spec_decode_metrics: None,
        weight_versions: None,
        routed_experts: None,
        timestamps: Default::default(),
    }
}

fn output_logprobs(wire: GenerateLogProbs) -> Logprobs {
    Logprobs {
        positions: wire
            .content
            .unwrap_or_default()
            .into_iter()
            .map(|row| {
                let mut entries = vec![TokenLogprob {
                    token_id: row.token_id,
                    logprob: row.logprob,
                    rank: row.rank.unwrap_or(0),
                }];
                entries.extend(
                    row.top_logprobs
                        .into_iter()
                        .filter(|entry| entry.token_id != row.token_id)
                        .map(|entry| TokenLogprob {
                            token_id: entry.token_id,
                            logprob: entry.logprob,
                            rank: entry.rank.unwrap_or(0),
                        }),
                );
                PositionLogprobs { entries }
            })
            .collect(),
    }
}
