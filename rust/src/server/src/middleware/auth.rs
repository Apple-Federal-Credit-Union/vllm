// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;
use std::task::{Context, Poll};

use axum::Json;
use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use futures::future::{self, Either, Ready};
use serde_json::json;
use tonic::Status;
use tower::Service;
use tower::layer::layer_fn;

use crate::state::{ApiKeyHash, AppState, hash_api_key};

const GUARDED_PREFIXES: &[&str] = &["/v1", "/v2", "/inference"];

/// gRPC service left open, like the HTTP `/health` route, so that Kubernetes
/// gRPC probes (which cannot send metadata) keep working.
const UNGUARDED_GRPC_PREFIX: &str = "/grpc.health.v1.Health/";

/// Authenticate guarded HTTP routes with an OpenAI-compatible bearer token.
///
/// Mirrors Python `AuthenticationMiddleware`: OPTIONS requests and non-guarded
/// helper endpoints such as `/health` are allowed through without a token.
pub async fn authenticate_api_key(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    if req.method() == Method::OPTIONS || !requires_auth(req.uri().path()) {
        return next.run(req).await;
    }

    if verify_token(req.headers().get(AUTHORIZATION), state.api_key_hashes()) {
        return next.run(req).await;
    }

    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "Unauthorized" })),
    )
        .into_response()
}

/// Return a Tower layer that authenticates gRPC calls with the same bearer
/// tokens as [`authenticate_api_key`], sent as `authorization` metadata.
///
/// Every service except the standard health service is guarded. Calls pass
/// through unchanged when no API keys are configured.
pub(crate) fn grpc_api_key_layer<S>(
    state: Arc<AppState>,
) -> impl tower::Layer<S, Service = GrpcApiKeyService<S>> + Clone {
    layer_fn(move |inner| GrpcApiKeyService {
        inner,
        state: state.clone(),
    })
}

/// Service produced by [`grpc_api_key_layer`].
#[derive(Clone)]
pub(crate) struct GrpcApiKeyService<S> {
    inner: S,
    state: Arc<AppState>,
}

type GrpcResponse = axum::http::Response<tonic::body::Body>;

impl<S, B> Service<axum::http::Request<B>> for GrpcApiKeyService<S>
where
    S: Service<axum::http::Request<B>, Response = GrpcResponse>,
{
    type Error = S::Error;
    type Future = Either<S::Future, Ready<Result<GrpcResponse, S::Error>>>;
    type Response = GrpcResponse;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: axum::http::Request<B>) -> Self::Future {
        if !self.state.has_api_keys()
            || req.uri().path().starts_with(UNGUARDED_GRPC_PREFIX)
            || verify_token(
                req.headers().get(AUTHORIZATION),
                self.state.api_key_hashes(),
            )
        {
            return Either::Left(self.inner.call(req));
        }
        Either::Right(future::ready(Ok(
            Status::unauthenticated("Unauthorized").into_http()
        )))
    }
}

fn requires_auth(path: &str) -> bool {
    GUARDED_PREFIXES.iter().any(|prefix| path.starts_with(prefix))
}

fn verify_token(authorization: Option<&HeaderValue>, api_key_hashes: &[ApiKeyHash]) -> bool {
    let Some(authorization) = authorization else {
        return false;
    };
    let Ok(authorization) = authorization.to_str() else {
        return false;
    };
    let Some((scheme, token)) = authorization.split_once(' ') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return false;
    }

    let token_hash = hash_api_key(token);
    let mut token_match = false;
    for api_key_hash in api_key_hashes {
        token_match |= constant_time_eq(&token_hash, api_key_hash);
    }
    token_match
}

fn constant_time_eq(left: &ApiKeyHash, right: &ApiKeyHash) -> bool {
    use subtle::ConstantTimeEq;

    bool::from(left.ct_eq(right))
}

#[cfg(test)]
mod tests {
    use super::constant_time_eq;
    use crate::state::hash_api_key;

    #[test]
    fn constant_time_eq_checks_sha256_digests() {
        assert!(constant_time_eq(
            &hash_api_key("secret"),
            &hash_api_key("secret")
        ));
        assert!(!constant_time_eq(
            &hash_api_key("secret"),
            &hash_api_key("secrex")
        ));
        assert!(!constant_time_eq(
            &hash_api_key("secret"),
            &hash_api_key("secret-more")
        ));
    }
}
