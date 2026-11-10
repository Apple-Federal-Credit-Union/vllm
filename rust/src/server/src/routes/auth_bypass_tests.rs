// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! API-key coverage for routes mounted outside `/v1`, `/v2`, and `/inference`.
//!
//! These tests call the real router. A rejection has to happen before the
//! handler tokenizes or the mock engine receives a utility call. The query
//! string on `/tokenize` is part of the request target but not of
//! `uri().path()`; a second lookup must not treat that request as unguarded.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use rmpv::Value;
use serial_test::serial;
use tower::Service as _;
use vllm_engine_core_client::mock_engine::default_ready_response;
use vllm_engine_core_client::protocol::decode_value;

use super::{
    boxed_test_future, recv_engine_message, send_outputs,
    test_admin_state_with_ready_and_engine_script, utility_none_result, utility_outputs,
    utility_result_value,
};
use crate::config::ApiServerOptions;

/// Stand-in configured on the test router. Not a deployed credential.
const PLACEHOLDER_API_KEY: &str = "placeholder-api-key";
/// Stand-in that must not match [`PLACEHOLDER_API_KEY`].
const REJECTED_API_KEY: &str = "placeholder-rejected";

const UNAUTHORIZED_BODY: &str = "{\"error\":\"Unauthorized\"}";

struct Exchange {
    status: StatusCode,
    body: String,
    engine_calls: Vec<String>,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn api_key_rejects_unprefixed_routes_before_tokenizer_or_engine_use() {
    let engine_log = Arc::new(Mutex::new(Vec::<String>::new()));
    let engine_log_for_script = Arc::clone(&engine_log);
    let (state, engine_task) = test_admin_state_with_ready_and_engine_script(
        default_ready_response(),
        move |dealer, push| {
            let engine_log = Arc::clone(&engine_log_for_script);
            boxed_test_future(async move {
                loop {
                    let message = recv_engine_message(dealer).await;
                    let record = describe_engine_message(&message);
                    engine_log.lock().expect("engine log").push(record);
                    if let Some(call_id) = utility_call_id(&message) {
                        let method = utility_method(&message).unwrap_or_default();
                        send_outputs(push, utility_outputs(call_id, reply_for(&method))).await;
                    }
                }
            })
        },
    )
    .await;
    let state = match Arc::try_unwrap(state) {
        Ok(state) => state,
        Err(_) => panic!("app state is unique"),
    };
    let state = Arc::new(
        state
            .with_api_keys(vec![PLACEHOLDER_API_KEY.to_string()])
            .with_profiler(Some("test-profiler".to_string()))
            .with_api_server_options(ApiServerOptions {
                enable_tokenizer_info_endpoint: true,
                ..Default::default()
            }),
    );
    let mut app = super::super::build_router_with_dev_mode(state, true);
    let mut failures = Vec::new();

    expect_status(
        &mut failures,
        &exchange(&mut app, &engine_log, "GET", "/v1/models", None, None).await,
        StatusCode::UNAUTHORIZED,
        "guarded /v1/models",
    );
    let health = exchange(&mut app, &engine_log, "GET", "/health", None, None).await;
    if health.status == StatusCode::UNAUTHORIZED {
        failures.push(format!("/health was rejected: {}", health.body));
    }
    expect_status(
        &mut failures,
        &exchange(
            &mut app,
            &engine_log,
            "POST",
            "/inference/v1/generate",
            None,
            Some("{}"),
        )
        .await,
        StatusCode::UNAUTHORIZED,
        "guarded /inference/v1/generate",
    );

    for bearer in [None, Some(REJECTED_API_KEY)] {
        let label = match bearer {
            None => "no Authorization header",
            Some(_) => "rejected bearer",
        };
        for (method, uri, body) in sensitive_requests() {
            let outcome = exchange(&mut app, &engine_log, method, uri, bearer, body).await;
            expect_rejected_before_use(&mut failures, &format!("{label} {method} {uri}"), &outcome);
        }
    }

    let queried = exchange(
        &mut app,
        &engine_log,
        "POST",
        "/tokenize?unused=1",
        None,
        Some(r#"{"model":"Qwen/Qwen1.5-0.5B-Chat","prompt":"hi","add_special_tokens":false}"#),
    )
    .await;
    expect_rejected_before_use(
        &mut failures,
        "no Authorization header POST /tokenize?unused=1",
        &queried,
    );
    if queried.body.contains("\"tokens\"") {
        failures.push(format!(
            "query-string /tokenize returned tokens before rejecting: {}",
            queried.body
        ));
    }

    let unauth_profile = exchange(
        &mut app,
        &engine_log,
        "POST",
        "/start_profile",
        None,
        Some("{}"),
    )
    .await;
    expect_rejected_before_use(
        &mut failures,
        "no Authorization header POST /start_profile",
        &unauth_profile,
    );
    let auth_profile = exchange(
        &mut app,
        &engine_log,
        "POST",
        "/start_profile",
        Some(PLACEHOLDER_API_KEY),
        Some("{}"),
    )
    .await;
    if auth_profile.status != StatusCode::OK
        || !auth_profile.engine_calls.iter().any(|call| call == "utility:profile")
    {
        failures.push(format!(
            "bearer /start_profile did not reach the engine after an unauthenticated \
             attempt (status {}, calls {:?}, body {}). An unauthenticated call must \
             not arm the profile session first.",
            auth_profile.status, auth_profile.engine_calls, auth_profile.body
        ));
    }

    let auth_tokenize = exchange(
        &mut app,
        &engine_log,
        "POST",
        "/tokenize",
        Some(PLACEHOLDER_API_KEY),
        Some(r#"{"model":"Qwen/Qwen1.5-0.5B-Chat","prompt":"hi","add_special_tokens":false}"#),
    )
    .await;
    if auth_tokenize.status != StatusCode::OK || !auth_tokenize.body.contains("\"tokens\"") {
        failures.push(format!(
            "bearer /tokenize did not return tokens: {} {}",
            auth_tokenize.status, auth_tokenize.body
        ));
    }

    let auth_detokenize = exchange(
        &mut app,
        &engine_log,
        "POST",
        "/detokenize",
        Some(PLACEHOLDER_API_KEY),
        Some(r#"{"model":"Qwen/Qwen1.5-0.5B-Chat","tokens":[104,105]}"#),
    )
    .await;
    if auth_detokenize.status != StatusCode::OK || !auth_detokenize.body.contains("\"prompt\"") {
        failures.push(format!(
            "bearer /detokenize did not return text: {} {}",
            auth_detokenize.status, auth_detokenize.body
        ));
    }

    let auth_rpc = exchange(
        &mut app,
        &engine_log,
        "POST",
        "/collective_rpc",
        Some(PLACEHOLDER_API_KEY),
        Some(r#"{"method":"caller_chosen_method"}"#),
    )
    .await;
    if auth_rpc.status != StatusCode::OK
        || !auth_rpc
            .engine_calls
            .iter()
            .any(|call| call == "collective_rpc:caller_chosen_method")
    {
        failures.push(format!(
            "bearer /collective_rpc did not forward the caller method: {} calls {:?} body {}",
            auth_rpc.status, auth_rpc.engine_calls, auth_rpc.body
        ));
    }

    let auth_scale = exchange(
        &mut app,
        &engine_log,
        "POST",
        "/scale_elastic_ep",
        Some(PLACEHOLDER_API_KEY),
        Some("{}"),
    )
    .await;
    if auth_scale.status == StatusCode::UNAUTHORIZED {
        failures.push(format!(
            "bearer /scale_elastic_ep was rejected: {}",
            auth_scale.body
        ));
    }

    drop(app);
    engine_task.abort_and_join().await;
    assert!(
        failures.is_empty(),
        "api-key gaps:\n{}",
        failures.join("\n")
    );
}

fn sensitive_requests() -> Vec<(&'static str, &'static str, Option<&'static str>)> {
    vec![
        (
            "POST",
            "/tokenize",
            Some(r#"{"model":"Qwen/Qwen1.5-0.5B-Chat","prompt":"hi","add_special_tokens":false}"#),
        ),
        (
            "POST",
            "/tokenize",
            Some(
                r#"{"model":"Qwen/Qwen1.5-0.5B-Chat","messages":[{"role":"user","content":"hi"}],"add_generation_prompt":false,"add_special_tokens":false}"#,
            ),
        ),
        (
            "POST",
            "/detokenize",
            Some(r#"{"model":"Qwen/Qwen1.5-0.5B-Chat","tokens":[104,105]}"#),
        ),
        ("POST", "/reset_prefix_cache", None),
        ("POST", "/reset_mm_cache", None),
        ("POST", "/reset_encoder_cache", None),
        (
            "POST",
            "/collective_rpc",
            Some(r#"{"method":"caller_chosen_method"}"#),
        ),
        (
            "POST",
            "/init_weight_transfer_engine",
            Some(r#"{"init_info":{}}"#),
        ),
        ("POST", "/shutdown_weight_transfer_engine", None),
        ("POST", "/start_weight_update", None),
        ("POST", "/start_draft_weight_update", None),
        (
            "POST",
            "/update_weights",
            Some(r#"{"update_info":{"kind":"test"}}"#),
        ),
        ("POST", "/finish_weight_update", Some("{}")),
        (
            "POST",
            "/update_weight_version",
            Some(r#"{"new_version":"next"}"#),
        ),
        ("GET", "/weight_info", None),
        ("POST", "/weight_checker", Some(r#"{"action":"checksum"}"#)),
        (
            "POST",
            "/abort_requests",
            Some(r#"{"request_ids":["not-inflight"]}"#),
        ),
        ("POST", "/sleep", None),
        ("POST", "/release_kv_cache_memory", None),
        ("POST", "/wake_up", None),
        ("GET", "/is_sleeping", None),
        ("POST", "/pause", None),
        ("POST", "/resume", None),
        ("GET", "/is_paused", None),
        ("GET", "/server_info", None),
        ("GET", "/get_world_size", None),
        ("POST", "/stop_profile", None),
        ("POST", "/scale_elastic_ep", Some("{}")),
        ("POST", "/is_scaling_elastic_ep", Some("{}")),
        ("GET", "/tokenizer_info", None),
    ]
}

fn expect_rejected_before_use(failures: &mut Vec<String>, label: &str, exchange: &Exchange) {
    if exchange.status != StatusCode::UNAUTHORIZED || exchange.body != UNAUTHORIZED_BODY {
        failures.push(format!(
            "{label} was not rejected before use: {} {} engine={:?}",
            exchange.status, exchange.body, exchange.engine_calls
        ));
    } else if !exchange.engine_calls.is_empty() {
        failures.push(format!(
            "{label} rejected only after the engine saw {:?}",
            exchange.engine_calls
        ));
    }
}

fn expect_status(failures: &mut Vec<String>, exchange: &Exchange, status: StatusCode, label: &str) {
    if exchange.status != status {
        failures.push(format!(
            "{label}: expected {status}, got {} {}",
            exchange.status, exchange.body
        ));
    }
}

async fn exchange(
    app: &mut axum::Router,
    engine_log: &Mutex<Vec<String>>,
    method: &str,
    uri: &str,
    bearer: Option<&str>,
    body: Option<&str>,
) -> Exchange {
    engine_log.lock().expect("engine log").clear();
    let mut builder = Request::builder().method(method).uri(uri);
    if body.is_some() {
        builder = builder.header("content-type", "application/json");
    }
    if let Some(bearer) = bearer {
        builder = builder.header("authorization", format!("Bearer {bearer}"));
    }
    let request = builder.body(Body::from(body.unwrap_or("").to_string())).expect("build request");
    let response = tokio::time::timeout(Duration::from_secs(5), app.call(request))
        .await
        .unwrap_or_else(|_| panic!("{method} {uri} timed out"))
        .unwrap_or_else(|error| panic!("{method} {uri} failed: {error}"));
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    Exchange {
        status,
        body: String::from_utf8_lossy(&bytes).into_owned(),
        engine_calls: std::mem::take(&mut *engine_log.lock().expect("engine log")),
    }
}

fn describe_engine_message(message: &[Bytes]) -> String {
    let Some(frame) = message.first() else {
        return "empty".to_string();
    };
    if frame.as_ref() == [0x01] {
        return "abort".to_string();
    }
    let Some(array) = utility_array(message) else {
        return format!("type:{:02x?}", frame.as_ref());
    };
    let method = array.get(2).and_then(Value::as_str).unwrap_or("?");
    if method == "collective_rpc" {
        let inner = array
            .get(3)
            .and_then(Value::as_array)
            .and_then(|args| args.first())
            .and_then(Value::as_str)
            .unwrap_or("?");
        format!("collective_rpc:{inner}")
    } else {
        format!("utility:{method}")
    }
}

fn utility_method(message: &[Bytes]) -> Option<String> {
    utility_array(message)
        .as_ref()
        .and_then(|array| array.get(2))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn utility_call_id(message: &[Bytes]) -> Option<u64> {
    if message.first().is_none_or(|frame| frame.as_ref() != [0x03]) {
        return None;
    }
    utility_array(message)
        .as_ref()
        .and_then(|array| array.get(1))
        .and_then(Value::as_u64)
}

fn utility_array(message: &[Bytes]) -> Option<Vec<Value>> {
    let payload = message.get(1)?;
    match decode_value(payload).ok()? {
        Value::Array(array) => Some(array),
        _ => None,
    }
}

fn reply_for(method: &str) -> vllm_engine_core_client::protocol::utility::UtilityResultEnvelope {
    match method {
        "collective_rpc" => utility_result_value(Vec::<String>::new()),
        "get_weight_version" => utility_result_value("v0"),
        "is_sleeping" | "wake_up" | "is_scheduler_paused" | "reset_prefix_cache" => {
            utility_result_value(true)
        }
        _ => utility_none_result(),
    }
}
