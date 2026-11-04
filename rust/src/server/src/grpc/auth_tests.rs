// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! API-key enforcement on the gRPC server built by [`crate::serve`].
//!
//! HTTP bearer checks live on the Axum router. These tests drive the same
//! `serve` path operators use with `--api-key` and a gRPC listener: every
//! Inference and Control RPC must reject a missing or non-matching bearer
//! before it submits work to the engine or cancels shutdown. Health stays
//! open. A bearer authorizes only the RPC that carries it.

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::os::fd::IntoRawFd;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rmpv::Value;
use serial_test::serial;
use socket2::{Domain, SockAddr, Socket, Type};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpStream;
use tokio_util::sync::CancellationToken;
use tonic::metadata::MetadataValue;
use tonic::transport::Endpoint;
use tonic::{Code, Status};
use tonic_health::pb::health_check_response::ServingStatus as HealthServingStatus;
use tonic_health::pb::{HealthCheckRequest, health_client::HealthClient};
use vllm_engine_core_client::mock_engine::{
    MockEngineConfig, connect_to_frontend, default_ready_response,
};
use vllm_engine_core_client::protocol::decode_value;
use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
    UtilityCallOutput,
};
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::utility::{UtilityOutput, UtilityResultEnvelope};
use vllm_engine_core_client::test_utils::IpcNamespace;
use vllm_engine_core_client::{EngineCoreClientConfig, EngineId};
use zeromq::prelude::{SocketRecv as _, SocketSend as _};
use zeromq::{PushSocket, ZmqMessage};

use super::pb;
use super::pb::control_client::ControlClient;
use super::pb::inference_client::InferenceClient;
use crate::{
    ApiServerOptions, ChatTemplateContentFormatOption, Config, CoordinatorMode, CorsConfig,
    GenerationConfigMode, HttpListenerMode, ParserSelection, RendererSelection,
};

/// Obvious placeholder configured on the server. Not a real credential.
const PLACEHOLDER_API_KEY: &str = "placeholder-api-key";
/// Second configured placeholder, distinct from [`PLACEHOLDER_API_KEY`].
const PLACEHOLDER_API_KEY_OTHER: &str = "placeholder-api-key-other";
const SERVED_MODEL: &str = "placeholder-model";
const AUTHORIZED_PROMPT_TOKEN: u32 = 1;
const OTHER_PROMPT_TOKEN: u32 = 2;
const SCRIPTED_OUTPUT_TOKEN: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
enum EngineEvent {
    Add {
        request_id: String,
        prompt_token_ids: Vec<u32>,
    },
    Abort {
        request_ids: Vec<String>,
    },
    Utility {
        method: String,
    },
}

fn bearer(token: &str) -> MetadataValue<tonic::metadata::Ascii> {
    format!("Bearer {token}").parse().expect("bearer is ascii metadata")
}

fn with_bearer<T>(request: impl tonic::IntoRequest<T>, token: &str) -> tonic::Request<T> {
    let mut request = request.into_request();
    request.metadata_mut().insert("authorization", bearer(token));
    request
}

fn generate_request(request_id: &str, prompt_token: u32) -> pb::GenerateRequest {
    pb::GenerateRequest {
        request_id: request_id.to_string(),
        model: SERVED_MODEL.to_string(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: vec![prompt_token],
        })),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens: 4,
            ..Default::default()
        }),
        response: Some(pb::ResponseOptions {
            output_text: Some(true),
            output_token_ids: true,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn reserve_tcp_listener() -> (String, i32) {
    let socket = Socket::new(Domain::IPV4, Type::STREAM, None).expect("tcp socket");
    socket
        .bind(&SockAddr::from(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)))
        .expect("bind loopback");
    let addr = socket.local_addr().expect("local addr").as_socket().expect("tcp addr");
    (addr.to_string(), socket.into_raw_fd())
}

fn write_model_dir() -> TempDir {
    let dir = TempDir::new().expect("model dir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"gpt2","vocab_size":8}"#,
    )
    .expect("config.json");
    std::fs::write(dir.path().join("tokenizer_config.json"), "{}").expect("tokenizer_config.json");
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "h": 1, "i": 2}
            }
        }"#,
    )
    .expect("tokenizer.json");
    dir
}

fn server_config(model_dir: &std::path::Path, http_fd: i32, grpc_fd: i32) -> (Config, String) {
    let ipc = IpcNamespace::new().expect("ipc namespace");
    let transport = EngineCoreClientConfig::new_single(ipc.handshake_endpoint())
        .with_model_name(SERVED_MODEL)
        .with_engine_stats_enabled(false)
        .with_local_input_output_addresses(Some(ipc.input_endpoint()), Some(ipc.output_endpoint()));
    let handshake = ipc.handshake_endpoint();
    // Keep the namespace alive for the process by leaking the temp dir. The
    // test process is short-lived; removing the sockets while ZMQ is connected
    // drops the engine session.
    std::mem::forget(ipc);
    let config = Config {
        transport_mode: transport.transport_mode,
        coordinator_mode: CoordinatorMode::None,
        model: model_dir.display().to_string(),
        revision: None,
        tokenizer: None,
        hf_config: None,
        hf_overrides: Default::default(),
        generation_config: GenerationConfigMode::default(),
        served_model_name: vec![SERVED_MODEL.to_string()],
        listener_mode: HttpListenerMode::InheritedFd { fd: http_fd },
        grpc_listener_mode: Some(HttpListenerMode::InheritedFd { fd: grpc_fd }),
        tool_call_parser: ParserSelection::Auto,
        reasoning_parser: ParserSelection::Auto,
        tool_strict_level: None,
        structured_outputs_backend: Default::default(),
        renderer: RendererSelection::Auto,
        language_model_only: true,
        chat_template: None,
        default_chat_template_kwargs: None,
        trust_request_chat_template: false,
        exclude_tools_when_tool_choice_none: false,
        limit_mm_per_prompt: Default::default(),
        lora_modules: Vec::new(),
        chat_template_content_format: ChatTemplateContentFormatOption::Auto,
        max_logprobs: None,
        stream_interval: std::num::NonZeroU32::MIN,
        api_server_options: ApiServerOptions::default(),
        cors: CorsConfig::default(),
        tls: None,
        api_keys: vec![
            PLACEHOLDER_API_KEY.to_string(),
            PLACEHOLDER_API_KEY_OTHER.to_string(),
        ],
        disable_log_stats: true,
        custom_histogram_buckets: Default::default(),
        log_stats_interval: None,
        disable_request_id_randomization: true,
        shutdown_timeout: Duration::from_secs(2),
        manages_engine: true,
        keep_alive_timeout: Duration::ZERO,
        profiler: None,
    };
    (config, handshake)
}

fn snapshot(events: &Mutex<Vec<EngineEvent>>) -> Vec<EngineEvent> {
    events.lock().expect("event log").clone()
}

async fn reject_without_engine_effect<T>(
    failures: &mut Vec<String>,
    events: &Mutex<Vec<EngineEvent>>,
    operation: &str,
    call: impl Future<Output = Result<T, Status>>,
) {
    let before = snapshot(events);
    let failure = match call.await {
        Ok(_) => Some(format!("{operation} completed without a matching bearer")),
        Err(status) if status.code() != Code::Unauthenticated => Some(format!(
            "{operation} returned {:?} instead of unauthenticated",
            status.code()
        )),
        Err(_) => None,
    };
    let after = snapshot(events);
    if after != before {
        failures.push(format!(
            "{operation} reached the engine before refusing: {}",
            format_engine_delta(&before, &after)
        ));
    }
    if let Some(failure) = failure {
        failures.push(failure);
    }
}

fn format_engine_delta(before: &[EngineEvent], after: &[EngineEvent]) -> String {
    after
        .iter()
        .skip(before.len())
        .map(|event| format!("{event:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn flush_auth_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "gRPC API key was not enforced:\n{}",
        failures.join("\n")
    );
}

async fn http_status(addr: &str, method: &str, path: &str, body: &str) -> u16 {
    let mut stream = TcpStream::connect(addr).await.expect("connect http");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await.expect("write http");
    let mut response = String::new();
    stream.read_to_string(&mut response).await.expect("read http");
    let status_line = response.lines().next().expect("status line");
    status_line
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("status number")
}

async fn wait_for_port(addr: &str, server: &mut tokio::task::JoinHandle<anyhow::Result<()>>) {
    for _ in 0..200 {
        if server.is_finished() {
            break;
        }
        if TcpStream::connect(addr).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    if server.is_finished() {
        match server.await {
            Ok(Err(error)) => panic!("serve failed before accepting: {error:#}"),
            other => panic!("serve exited before accepting: {other:?}"),
        }
    }
    panic!("timed out waiting for {addr}");
}

fn scripted_output(request_id: &str) -> EngineCoreOutputs {
    RequestBatchOutputs {
        outputs: vec![EngineCoreOutput {
            request_id: request_id.to_string(),
            new_token_ids: vec![SCRIPTED_OUTPUT_TOKEN],
            finish_reason: Some(EngineCoreFinishReason::Stop),
            ..Default::default()
        }],
        ..Default::default()
    }
    .into()
}

async fn send_outputs(push: &mut PushSocket, outputs: EngineCoreOutputs) {
    push.send(ZmqMessage::from(
        rmp_serde::to_vec_named(&outputs).expect("encode outputs"),
    ))
    .await
    .expect("send outputs");
}

fn utility_result(method: &str) -> Value {
    match method {
        "is_scheduler_paused" | "is_sleeping" | "add_lora" | "remove_lora" | "wake_up" => {
            Value::Boolean(true)
        }
        "get_weight_version" => Value::from("placeholder-weight-version"),
        "collective_rpc" => Value::Array(vec![Value::Nil]),
        _ => Value::Nil,
    }
}

fn record_utility(events: &Mutex<Vec<EngineEvent>>, fields: &[Value]) -> u64 {
    let call_id = fields[1].as_u64().expect("utility call id");
    let method = fields[2].as_str().unwrap_or("").to_string();
    let method = if method == "collective_rpc" {
        fields
            .get(3)
            .and_then(Value::as_array)
            .and_then(|args| args.first())
            .and_then(Value::as_str)
            .unwrap_or("collective_rpc")
            .to_string()
    } else {
        method
    };
    events.lock().expect("event log").push(EngineEvent::Utility { method });
    call_id
}

async fn serve_mock_engine(handshake: String, events: Arc<Mutex<Vec<EngineEvent>>>) {
    let mut ready = default_ready_response();
    ready.supports_lora = true;
    ready.max_loras = 2;
    ready.enable_sleep_mode = true;
    ready.weight_transfer_backend = Some("nccl".to_string());
    ready.supports_draft_weight_updates = true;
    let sockets = connect_to_frontend(
        handshake,
        EngineId::from_engine_index(0),
        MockEngineConfig {
            local: true,
            headless: true,
            ready_response: ready,
            connect_timeout: Duration::from_secs(60),
        },
    )
    .await
    .expect("mock engine handshake");
    let mut sockets = sockets.data_sockets.into_iter();
    let Some(data) = sockets.next() else {
        panic!("mock engine missing data sockets");
    };
    let mut dealer = data.dealer;
    let mut push = data.push;

    loop {
        let Ok(message) = dealer.recv().await else {
            break;
        };
        let frames = message.into_vec();
        let kind = frames.first().map(|frame| frame.as_ref()).unwrap_or_default();
        match kind {
            [0x00] => {
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&frames[1]).expect("decode add");
                let request_id = request.request_id.clone();
                events.lock().expect("event log").push(EngineEvent::Add {
                    request_id: request_id.clone(),
                    prompt_token_ids: request.prompt_token_ids.unwrap_or_default(),
                });
                if !request_id.starts_with("held-") {
                    send_outputs(&mut push, scripted_output(&request_id)).await;
                }
            }
            [0x01] => {
                let request_ids: Vec<String> =
                    rmp_serde::from_slice(&frames[1]).expect("decode abort");
                events.lock().expect("event log").push(EngineEvent::Abort { request_ids });
            }
            [0x03] => {
                let payload = decode_value(&frames[1]).expect("decode utility");
                let fields = payload.as_array().expect("utility payload");
                let method = fields[2].as_str().unwrap_or("");
                let call_id = record_utility(&events, fields);
                send_outputs(
                    &mut push,
                    UtilityCallOutput {
                        output: UtilityOutput {
                            call_id: call_id.into(),
                            failure_message: None,
                            result: Some(UtilityResultEnvelope::without_type_info(utility_result(
                                method,
                            ))),
                        },
                        ..Default::default()
                    }
                    .into(),
                )
                .await;
            }
            _ => {}
        }
    }
}

async fn assert_output_token(response: pb::GenerateResponse, request_id: &str) {
    let outputs = response.outputs.expect("outputs");
    assert_eq!(
        outputs.token_ids,
        vec![SCRIPTED_OUTPUT_TOKEN],
        "{request_id} must return the token produced for that request"
    );
    let finish = outputs.finish_info.expect("finish");
    assert_eq!(
        finish.finish_reason,
        pb::finish_info::FinishReason::Stop as i32
    );
}

async fn stream_still_open(stream: &mut tonic::Streaming<pb::GenerateResponse>, request_id: &str) {
    let polled = tokio::time::timeout(Duration::from_millis(200), stream.message()).await;
    assert!(
        polled.is_err(),
        "{request_id} must still be running; abort of a different request must not finish it, got {polled:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial]
async fn grpc_api_key_gates_every_inference_and_control_rpc() {
    let model_dir = write_model_dir();
    let (http_addr, http_fd) = reserve_tcp_listener();
    let (grpc_addr, grpc_fd) = reserve_tcp_listener();
    let (config, handshake) = server_config(model_dir.path(), http_fd, grpc_fd);
    let events = Arc::new(Mutex::new(Vec::new()));
    let engine_events = Arc::clone(&events);
    let engine_task =
        tokio::spawn(async move { serve_mock_engine(handshake, engine_events).await });

    let shutdown = CancellationToken::new();
    let server_shutdown = shutdown.clone();
    let mut server_task = tokio::spawn(async move { crate::serve(config, server_shutdown).await });
    wait_for_port(&http_addr, &mut server_task).await;
    wait_for_port(&grpc_addr, &mut server_task).await;

    let chat_body = r#"{"model":"placeholder-model","messages":[{"role":"user","content":"hi"}]}"#;
    let events_before_http = snapshot(&events);
    let chat_status = http_status(&http_addr, "POST", "/v1/chat/completions", chat_body).await;
    assert_eq!(
        chat_status, 401,
        "HTTP generation without a bearer must be rejected"
    );
    assert_eq!(
        snapshot(&events),
        events_before_http,
        "rejected HTTP generation must not reach the engine"
    );
    assert_eq!(http_status(&http_addr, "GET", "/health", "").await, 200);

    let channel = Endpoint::from_shared(format!("http://{grpc_addr}"))
        .expect("endpoint")
        .connect_timeout(Duration::from_secs(5))
        .connect()
        .await
        .expect("grpc channel");
    let mut inference = InferenceClient::new(channel.clone());
    let mut control = ControlClient::new(channel.clone());
    let mut health = HealthClient::new(channel.clone());

    let mut failures = Vec::new();

    for service in ["", "vllm.Inference", "vllm.Control"] {
        let status = health
            .check(HealthCheckRequest {
                service: service.to_string(),
            })
            .await
            .unwrap_or_else(|error| panic!("health {service} without a bearer: {error}"))
            .into_inner();
        assert_eq!(
            status.status,
            HealthServingStatus::Serving as i32,
            "{service}"
        );
    }

    reject_without_engine_effect(
        &mut failures,
        &events,
        "Generate",
        inference.generate(generate_request("unauth-generate", OTHER_PROMPT_TOKEN)),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "GenerateStream",
        inference.generate_stream(generate_request("unauth-stream", OTHER_PROMPT_TOKEN)),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "GetServerInfo",
        control.get_server_info(pb::GetServerInfoRequest {}),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "GetModelInfo",
        control.get_model_info(pb::GetModelInfoRequest {}),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "Abort",
        control.abort(pb::AbortRequest {
            request_ids: vec!["not-running".to_string()],
        }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "LoadLora",
        control.load_lora(pb::LoadLoraRequest {
            lora_name: "adapter-alpha".to_string(),
            source_path: "org/adapter-alpha".to_string(),
        }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "UnloadLora",
        control.unload_lora(pb::UnloadLoraRequest {
            lora_name: "adapter-alpha".to_string(),
        }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "ListLoras",
        control.list_loras(pb::ListLorasRequest {}),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "GetKvEventSources",
        control.get_kv_event_sources(pb::GetKvEventSourcesRequest {}),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "PauseGeneration",
        control.pause_generation(pb::PauseGenerationRequest {
            mode: pb::PauseMode::Abort as i32,
            clear_cache: Some(true),
        }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "ResumeGeneration",
        control.resume_generation(pb::ResumeGenerationRequest {}),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "IsPaused",
        control.is_paused(pb::IsPausedRequest {}),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "Sleep",
        control.sleep(pb::SleepRequest {
            level: Some(1),
            mode: pb::PauseMode::Abort as i32,
        }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "WakeUp",
        control.wake_up(pb::WakeUpRequest { tags: vec![] }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "IsSleeping",
        control.is_sleeping(pb::IsSleepingRequest {}),
    )
    .await;
    let weight_json = br#"{"names":["model.weight"]}"#.to_vec();
    reject_without_engine_effect(
        &mut failures,
        &events,
        "InitWeightTransferEngine",
        control.init_weight_transfer_engine(pb::InitWeightTransferEngineRequest {
            init_info_json: weight_json.clone(),
        }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "StartWeightUpdate",
        control.start_weight_update(pb::StartWeightUpdateRequest {}),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "StartDraftWeightUpdate",
        control.start_draft_weight_update(pb::StartDraftWeightUpdateRequest {}),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "UpdateWeights",
        control.update_weights(pb::UpdateWeightsRequest {
            update_info_json: weight_json,
        }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "FinishWeightUpdate",
        control.finish_weight_update(pb::FinishWeightUpdateRequest {
            weight_version: Some("placeholder-weight-version".to_string()),
        }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "UpdateWeightVersion",
        control.update_weight_version(pb::UpdateWeightVersionRequest {
            weight_version: "placeholder-weight-version".to_string(),
        }),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "GetWeightVersion",
        control.get_weight_version(pb::GetWeightVersionRequest {}),
    )
    .await;

    reject_without_engine_effect(
        &mut failures,
        &events,
        "Generate with a non-matching bearer",
        inference.generate(with_bearer(
            generate_request("wrong-bearer", OTHER_PROMPT_TOKEN),
            "placeholder-api-key-wrong",
        )),
    )
    .await;
    reject_without_engine_effect(
        &mut failures,
        &events,
        "Generate with a prefix of a configured key",
        inference.generate(with_bearer(
            generate_request("prefix-bearer", OTHER_PROMPT_TOKEN),
            "placeholder",
        )),
    )
    .await;
    {
        let mut request = tonic::Request::new(generate_request("side-channel", OTHER_PROMPT_TOKEN));
        request
            .metadata_mut()
            .insert("authorization", bearer("placeholder-api-key-wrong"));
        request.metadata_mut().insert("x-api-key", bearer(PLACEHOLDER_API_KEY));
        reject_without_engine_effect(
            &mut failures,
            &events,
            "Generate authorized by a side-channel header",
            inference.generate(request),
        )
        .await;
    }

    let shutdown_was_cancelled = shutdown.is_cancelled();
    reject_without_engine_effect(
        &mut failures,
        &events,
        "Shutdown",
        control.shutdown(pb::ShutdownRequest {}),
    )
    .await;
    if shutdown.is_cancelled() && !shutdown_was_cancelled {
        failures
            .push("Shutdown cancelled the server token before the bearer was accepted".to_string());
    }
    flush_auth_failures(&failures);

    let before_authorized = snapshot(&events);
    let authorized = inference
        .generate(with_bearer(
            generate_request("authorized-generate", AUTHORIZED_PROMPT_TOKEN),
            PLACEHOLDER_API_KEY,
        ))
        .await
        .expect("matching bearer must allow Generate")
        .into_inner();
    assert_output_token(authorized, "authorized-generate").await;
    let authorized_events = &snapshot(&events)[before_authorized.len()..];
    assert_eq!(
        authorized_events,
        &[EngineEvent::Add {
            request_id: "authorized-generate".to_string(),
            prompt_token_ids: vec![AUTHORIZED_PROMPT_TOKEN],
        }],
        "the engine must run the prompt from the request that presented the bearer"
    );

    let before_other = snapshot(&events);
    let other = inference
        .generate(with_bearer(
            generate_request("other-key-generate", OTHER_PROMPT_TOKEN),
            PLACEHOLDER_API_KEY_OTHER,
        ))
        .await
        .expect("the other configured key must allow its own Generate")
        .into_inner();
    assert_output_token(other, "other-key-generate").await;
    assert_eq!(
        &snapshot(&events)[before_other.len()..],
        &[EngineEvent::Add {
            request_id: "other-key-generate".to_string(),
            prompt_token_ids: vec![OTHER_PROMPT_TOKEN],
        }]
    );

    reject_without_engine_effect(
        &mut failures,
        &events,
        "Generate after a successful call, without a bearer",
        inference.generate(generate_request("after-success", OTHER_PROMPT_TOKEN)),
    )
    .await;
    flush_auth_failures(&failures);

    let before_concurrent = snapshot(&events);
    let mut concurrent_authorized = InferenceClient::new(channel.clone());
    let mut concurrent_unauthorized = InferenceClient::new(channel.clone());
    let (authorized_result, unauthorized_result) = tokio::join!(
        concurrent_authorized.generate(with_bearer(
            generate_request("concurrent-auth", AUTHORIZED_PROMPT_TOKEN),
            PLACEHOLDER_API_KEY,
        )),
        concurrent_unauthorized.generate(generate_request("concurrent-other", OTHER_PROMPT_TOKEN)),
    );
    let unauthorized_status = unauthorized_result.expect_err("concurrent unauthenticated Generate");
    assert_eq!(unauthorized_status.code(), Code::Unauthenticated);
    assert_output_token(
        authorized_result.expect("concurrent authorized Generate").into_inner(),
        "concurrent-auth",
    )
    .await;
    assert_eq!(
        &snapshot(&events)[before_concurrent.len()..],
        &[EngineEvent::Add {
            request_id: "concurrent-auth".to_string(),
            prompt_token_ids: vec![AUTHORIZED_PROMPT_TOKEN],
        }],
        "a concurrent unauthenticated prompt must not be the one submitted"
    );

    let mut held_a = inference
        .generate_stream(with_bearer(
            generate_request("held-a", AUTHORIZED_PROMPT_TOKEN),
            PLACEHOLDER_API_KEY,
        ))
        .await
        .expect("start held-a")
        .into_inner();
    let mut held_b = inference
        .generate_stream(with_bearer(
            generate_request("held-b", OTHER_PROMPT_TOKEN),
            PLACEHOLDER_API_KEY,
        ))
        .await
        .expect("start held-b")
        .into_inner();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let seen = snapshot(&events);
            let has_a = seen.iter().any(|event| {
                matches!(event, EngineEvent::Add { request_id, .. } if request_id == "held-a")
            });
            let has_b = seen.iter().any(|event| {
                matches!(event, EngineEvent::Add { request_id, .. } if request_id == "held-b")
            });
            if has_a && has_b {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("held streams must reach the engine");

    let before_bad_abort = snapshot(&events);
    reject_without_engine_effect(
        &mut failures,
        &events,
        "Abort of an in-flight request",
        control.abort(pb::AbortRequest {
            request_ids: vec!["held-b".to_string()],
        }),
    )
    .await;
    flush_auth_failures(&failures);
    assert_eq!(snapshot(&events), before_bad_abort);
    stream_still_open(&mut held_a, "held-a").await;
    stream_still_open(&mut held_b, "held-b").await;

    let aborted = control
        .abort(with_bearer(
            pb::AbortRequest {
                request_ids: vec!["held-b".to_string()],
            },
            PLACEHOLDER_API_KEY,
        ))
        .await
        .expect("matching bearer must allow Abort of the named request");
    assert_eq!(aborted.into_inner(), pb::AbortResponse {});
    let abort_events: Vec<_> = snapshot(&events)
        .into_iter()
        .filter(|event| matches!(event, EngineEvent::Abort { .. }))
        .collect();
    assert_eq!(
        abort_events,
        vec![EngineEvent::Abort {
            request_ids: vec!["held-b".to_string()],
        }],
        "Abort must cancel only the request id in the authorized RPC"
    );
    let held_b_end = tokio::time::timeout(Duration::from_secs(2), held_b.message())
        .await
        .expect("held-b should finish after its authorized abort")
        .expect("held-b stream");
    assert!(held_b_end.is_some(), "held-b must observe the abort");
    stream_still_open(&mut held_a, "held-a after held-b abort").await;

    control
        .load_lora(with_bearer(
            pb::LoadLoraRequest {
                lora_name: "adapter-alpha".to_string(),
                source_path: "org/adapter-alpha".to_string(),
            },
            PLACEHOLDER_API_KEY,
        ))
        .await
        .expect("matching bearer must allow LoadLora");
    control
        .load_lora(with_bearer(
            pb::LoadLoraRequest {
                lora_name: "adapter-beta".to_string(),
                source_path: "org/adapter-beta".to_string(),
            },
            PLACEHOLDER_API_KEY_OTHER,
        ))
        .await
        .expect("the other configured key must allow LoadLora of its own adapter");
    reject_without_engine_effect(
        &mut failures,
        &events,
        "UnloadLora of a loaded adapter",
        control.unload_lora(pb::UnloadLoraRequest {
            lora_name: "adapter-alpha".to_string(),
        }),
    )
    .await;
    flush_auth_failures(&failures);
    let listed = control
        .list_loras(with_bearer(pb::ListLorasRequest {}, PLACEHOLDER_API_KEY))
        .await
        .expect("matching bearer must allow ListLoras")
        .into_inner();
    let names: Vec<_> = listed.adapters.iter().map(|adapter| adapter.lora_name.as_str()).collect();
    assert_eq!(names, ["adapter-alpha", "adapter-beta"]);

    control
        .unload_lora(with_bearer(
            pb::UnloadLoraRequest {
                lora_name: "adapter-alpha".to_string(),
            },
            PLACEHOLDER_API_KEY,
        ))
        .await
        .expect("matching bearer must unload the named adapter");
    let listed = control
        .list_loras(with_bearer(
            pb::ListLorasRequest {},
            PLACEHOLDER_API_KEY_OTHER,
        ))
        .await
        .expect("list after unload")
        .into_inner();
    let names: Vec<_> = listed.adapters.iter().map(|adapter| adapter.lora_name.as_str()).collect();
    assert_eq!(
        names,
        ["adapter-beta"],
        "UnloadLora must remove the adapter named in the authorized request"
    );

    let before_pause = snapshot(&events);
    control
        .pause_generation(with_bearer(
            pb::PauseGenerationRequest {
                mode: pb::PauseMode::Abort as i32,
                clear_cache: Some(false),
            },
            PLACEHOLDER_API_KEY,
        ))
        .await
        .expect("matching bearer must allow PauseGeneration");
    assert_eq!(
        &snapshot(&events)[before_pause.len()..],
        &[EngineEvent::Utility {
            method: "pause_generation".to_string(),
        }]
    );
    let before_weights = snapshot(&events);
    control
        .update_weights(with_bearer(
            pb::UpdateWeightsRequest {
                update_info_json: br#"{"names":["model.weight"]}"#.to_vec(),
            },
            PLACEHOLDER_API_KEY,
        ))
        .await
        .expect("matching bearer must allow UpdateWeights");
    assert_eq!(
        &snapshot(&events)[before_weights.len()..],
        &[EngineEvent::Utility {
            method: "update_weights".to_string(),
        }]
    );

    let info = control
        .get_server_info(with_bearer(
            pb::GetServerInfoRequest {},
            PLACEHOLDER_API_KEY,
        ))
        .await
        .expect("matching bearer must allow GetServerInfo")
        .into_inner();
    assert_eq!(info.engine_version, "test-vllm-version");

    assert!(!shutdown.is_cancelled());
    reject_without_engine_effect(
        &mut failures,
        &events,
        "Shutdown",
        control.shutdown(pb::ShutdownRequest {}),
    )
    .await;
    assert!(
        !shutdown.is_cancelled(),
        "Shutdown without a bearer must not cancel the server shutdown token"
    );
    let still_healthy = health
        .check(HealthCheckRequest {
            service: "vllm.Inference".to_string(),
        })
        .await
        .expect("server must still be serving after rejected Shutdown")
        .into_inner();
    assert_eq!(still_healthy.status, HealthServingStatus::Serving as i32);
    flush_auth_failures(&failures);

    control
        .shutdown(with_bearer(pb::ShutdownRequest {}, PLACEHOLDER_API_KEY))
        .await
        .expect("matching bearer must allow Shutdown when this process owns the engine");
    assert!(
        shutdown.is_cancelled(),
        "authorized Shutdown must cancel the token passed to serve"
    );

    engine_task.abort();
    let _ = tokio::time::timeout(Duration::from_secs(5), server_task).await;
}
