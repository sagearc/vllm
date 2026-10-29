// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use serde_json::json;
use vllm_text::{Prompt, SamplingHints, TextBackend, TextRequest, TextRequestProcessor};
use vllm_tokenizer::test_utils::TestTokenizer;

use super::*;

struct Backend;

impl TextBackend for Backend {
    fn model_id(&self) -> &str {
        "test-model"
    }
    fn tokenizer(&self) -> vllm_tokenizer::DynTokenizer {
        Arc::new(TestTokenizer::new())
    }
    fn sampling_hints(&self) -> vllm_text::Result<SamplingHints> {
        Ok(SamplingHints {
            default_temperature: Some(0.7),
            default_max_tokens: Some(12),
            primary_eos_token_id: Some(2),
            ..Default::default()
        })
    }
}

fn processor() -> TextRequestProcessor {
    TextRequestProcessor::new(Arc::new(Backend), 128)
}

fn prepared() -> GenerationRequest {
    let mut request = TextRequest {
        request_id: "chat-test".into(),
        prompt: Prompt::TokenIds(vec![10, 20]),
        ..TextRequest::for_test()
    };
    request.sampling_params.max_tokens = None;
    request.sampling_params.ignore_eos = true;
    request.sampling_params.bad_words = Some(vec!["hi".into()]);
    request.sampling_params.vllm_xargs = Some(
        [
            ("custom".into(), json!(7)),
            ("kv_transfer_params".into(), json!({"remote_decode": true})),
        ]
        .into(),
    );
    TextRequestProcessor::new(Arc::new(Backend), 128)
        .prepare_generation(request)
        .unwrap()
        .generate_request
}

#[test]
fn http_request_preserves_resolved_defaults_and_public_sampling_semantics() {
    let p = prepared();
    let body = request_body("test-model", NonZeroU32::MIN, &p).unwrap();
    assert_eq!(body["token_ids"], json!([10, 20]));
    assert_eq!(body["sampling_params"]["max_tokens"], 12);
    assert!((body["sampling_params"]["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
    assert_eq!(body["sampling_params"]["ignore_eos"], true);
    assert_eq!(body["sampling_params"]["bad_words"], json!(["hi"]));
    assert!(body["sampling_params"].get("bad_words_token_ids").is_none());
    assert_eq!(body["sampling_params"]["extra_args"], json!({"custom": 7}));
    assert_eq!(body["kv_transfer_params"], json!({"remote_decode": true}));
    assert!(body["sampling_params"].get("_eos_token_id").is_none());
    // The Rust HTTP endpoint accepts the same Python-compatible extension spelling.
    let wire: HttpGenerateRequest = serde_json::from_value(body).unwrap();
    assert_eq!(wire.sampling_params.inner.vllm_xargs.unwrap()["custom"], 7);
}

#[test]
fn logprobs_keep_sampled_token_first_without_duplicating_it() {
    let wire = serde_json::from_value(json!({"content": [{
        "token_id": 20, "logprob": -1.7, "rank": 2,
        "top_logprobs": [
            {"token_id": 10, "logprob": -0.3, "rank": 1},
            {"token_id": 20, "logprob": -1.7, "rank": 2}
        ]
    }]}))
    .unwrap();
    let probabilities = output_logprobs(wire);
    let entries = &probabilities.positions[0].entries;
    assert_eq!(
        entries.iter().map(|p| (p.token_id, p.logprob, p.rank)).collect::<Vec<_>>(),
        vec![(20, -1.7, 2), (10, -0.3, 1)]
    );
}

#[test]
fn selective_logprobs_are_rejected_before_submission() {
    let mut request = TextRequest::for_test();
    request.prompt = Prompt::TokenIds(vec![10, 20]);
    request.sampling_params.logprob_token_ids = Some(vec![30, 40]);
    let p = TextRequestProcessor::new(Arc::new(Backend), 128)
        .prepare_generation(request)
        .unwrap()
        .generate_request;
    let error = request_body("test-model", NonZeroU32::MIN, &p).unwrap_err();
    assert_eq!(http_error(&error).status, 400);
    let error = crate::error::text_submit_error("submit", error.into());
    let response = error.to_error_response();
    assert_eq!(response.error.error_type, "BadRequestError");
    assert!(response.error.message.contains("selective logprobs"));
}

async fn endpoint(
    body: axum::body::Body,
    status: u16,
) -> (
    String,
    tokio::sync::oneshot::Receiver<Value>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (sent, received) = tokio::sync::oneshot::channel();
    let response = Arc::new(Mutex::new(Some((sent, body))));
    let app = axum::Router::new().route(
        "/inference/v1/generate",
        axum::routing::post(move |axum::Json(request): axum::Json<Value>| {
            let (sent, body) = response.lock().unwrap().take().unwrap();
            async move {
                sent.send(request).unwrap();
                axum::http::Response::builder()
                    .status(status)
                    .header("content-type", "text/event-stream")
                    .body(body)
                    .unwrap()
            }
        }),
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, received, task)
}

fn fragmented_body(data: &str) -> axum::body::Body {
    let pieces: Vec<_> = data
        .as_bytes()
        .chunks(3)
        .map(|bytes| Ok::<_, std::convert::Infallible>(bytes.to_vec()))
        .collect();
    axum::body::Body::from_stream(futures::stream::iter(pieces))
}

#[tokio::test]
async fn fragmented_stream_keeps_final_usage_and_token_metadata() {
    let data = concat!(
        "data: {\"choices\":[{\"index\":0,\"token_ids\":[30]}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"token_ids\":[40],\"finish_reason\":\"stop\",\"stop_reason\":2}],\"kv_transfer_params\":{\"remote_decode\":true}}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\n",
        "data: [DONE]\n\n",
    );
    let (url, received, server) = endpoint(fragmented_body(data), 200).await;
    let client = EngineHTTPClient::new(&url, "test-model".into(), NonZeroU32::MIN, 128).unwrap();
    let p = prepared();
    let mut stream = client.generate(p, &processor()).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.token_ids, vec![30]);
    assert_eq!(&*first.prompt_info.unwrap().prompt_token_ids, &[10, 20]);
    let last = stream.next().await.unwrap().unwrap();
    assert_eq!(last.token_ids, vec![40]);
    assert!(matches!(last.finish_reason, Some(FinishReason::Stop(_))));
    assert_eq!(last.cached_token_count, 1);
    assert_eq!(
        last.kv_transfer_params.as_deref(),
        Some(&json!({"remote_decode": true}))
    );
    assert!(stream.next().await.is_none());
    let request = received.await.unwrap();
    assert_eq!(request["output_mode"], "tokens");
    assert_eq!(request["stream"], true);
    assert!(client.active.lock().unwrap().is_empty());
    server.abort();
}

#[tokio::test]
async fn stream_truncation_and_upstream_errors_are_not_success() {
    for data in [
        "data: {\"choices\":[{\"index\":0,\"token_ids\":[],\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
        "data: {\"error\":{\"message\":\"engine failed\"}}\n\n",
        "data: {\"error\":{\"message\":\"engine failed\"},\"choices\":[]}\n\n",
    ] {
        let (url, received, server) = endpoint(fragmented_body(data), 200).await;
        let client =
            EngineHTTPClient::new(&url, "test-model".into(), NonZeroU32::MIN, 128).unwrap();
        let p = prepared();
        let mut stream = client.generate(p, &processor()).await.unwrap();
        let error = stream.next().await.unwrap().unwrap_err();
        assert_eq!(http_error(&error).status, 502);
        if data.contains("engine failed") {
            assert!(http_error(&error).message.contains("engine failed"));
        }
        received.await.unwrap();
        server.abort();
    }
    let (url, received, server) = endpoint(axum::body::Body::from("overloaded"), 429).await;
    let client = EngineHTTPClient::new(&url, "test-model".into(), NonZeroU32::MIN, 128).unwrap();
    let p = prepared();
    let Err(error) = client.generate(p, &processor()).await else {
        panic!("expected rejection")
    };
    let error = crate::error::chat_submit_error("chat failed", vllm_chat::Error::Text(error));
    assert_eq!(
        error.status_code(),
        axum::http::StatusCode::TOO_MANY_REQUESTS
    );
    received.await.unwrap();
    server.abort();
}

#[tokio::test]
async fn upstream_validation_preserves_error_details() {
    let body = json!({"error": {
        "message": "top_p must be within (0, 1]", "type": "BadRequestError",
        "param": "top_p", "code": 400
    }});
    let (url, received, server) = endpoint(axum::body::Body::from(body.to_string()), 400).await;
    let client = EngineHTTPClient::new(&url, "test-model".into(), NonZeroU32::MIN, 128).unwrap();
    let p = prepared();
    let Err(error) = client.generate(p, &processor()).await else {
        panic!("expected validation failure")
    };
    let error = crate::error::text_submit_error("submit", error);
    assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
    let response = error.to_error_response();
    assert_eq!(response.error.message, "top_p must be within (0, 1]");
    assert_eq!(response.error.error_type, "BadRequestError");
    assert_eq!(response.error.param.as_deref(), Some("top_p"));
    assert_eq!(response.error.code, 400);
    received.await.unwrap();
    server.abort();
}

#[tokio::test]
async fn abort_and_stream_drop_close_the_upstream_connection() {
    struct Dropped(Option<tokio::sync::oneshot::Sender<()>>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            if let Some(sent) = self.0.take() {
                let _ = sent.send(());
            }
        }
    }
    for abort in [false, true] {
        let (sent, dropped) = tokio::sync::oneshot::channel();
        let guard = Dropped(Some(sent));
        let first = futures::stream::once(async {
            Ok::<_, std::convert::Infallible>(
                "data: {\"choices\":[{\"index\":0,\"token_ids\":[30]}]}\n\n",
            )
        });
        let pending = futures::stream::once(async move {
            let _guard = guard;
            std::future::pending().await
        });
        let (url, received, server) =
            endpoint(axum::body::Body::from_stream(first.chain(pending)), 200).await;
        let client =
            EngineHTTPClient::new(&url, "test-model".into(), NonZeroU32::MIN, 128).unwrap();
        let p = prepared();
        let mut stream = client.generate(p, &processor()).await.unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap().token_ids, vec![30]);
        assert_eq!(client.active.lock().unwrap().len(), 1);
        if abort {
            client.abort(&["chat-test".into()]).await.unwrap();
            let output = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert!(matches!(output.finish_reason, Some(FinishReason::Abort)));
        }
        drop(stream);
        tokio::time::timeout(std::time::Duration::from_secs(5), dropped)
            .await
            .unwrap()
            .unwrap();
        assert!(client.active.lock().unwrap().is_empty());
        received.await.unwrap();
        server.abort();
    }
}

fn http_error(error: &Error) -> &HttpGenerationError {
    let Error::Backend { source } = error else {
        panic!("expected backend failure")
    };
    source.downcast_ref().expect("HTTP failure")
}
