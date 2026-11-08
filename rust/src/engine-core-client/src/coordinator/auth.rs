// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;

use bytes::Bytes;
use hmac::{Hmac, Mac};
use sha2::Sha256;

/// MAC domain for engine-to-coordinator messages.
///
/// Must match `_ENGINE_OUTPUT_DOMAIN` in Python `vllm/v1/engine/coordinator.py`.
const ENGINE_OUTPUT_DOMAIN: &[u8] = b"vllm-dp-coordinator/engine-output";

/// Key that authenticates engine messages sent to the coordinator output
/// socket. Engines receive it in the startup handshake and append
/// `HMAC-SHA256(key, domain || (len_le_u64(frame) || frame)*)` as a final frame.
#[derive(Clone)]
pub(crate) struct CoordinatorAuthKey(Vec<u8>);

impl CoordinatorAuthKey {
    /// Generate a fresh random key.
    pub(crate) fn generate() -> Self {
        Self(rand::random::<[u8; 32]>().to_vec())
    }

    pub(crate) fn from_bytes(key: Vec<u8>) -> Self {
        Self(key)
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    fn mac(&self, frames: &[Bytes]) -> Hmac<Sha256> {
        let mut mac =
            Hmac::<Sha256>::new_from_slice(&self.0).expect("HMAC accepts keys of any length");
        mac.update(ENGINE_OUTPUT_DOMAIN);
        for frame in frames {
            mac.update(&(frame.len() as u64).to_le_bytes());
            mac.update(frame);
        }
        mac
    }

    /// Compute the MAC frame to append to an engine message.
    pub(crate) fn sign(&self, frames: &[Bytes]) -> Bytes {
        Bytes::from(self.mac(frames).finalize().into_bytes().to_vec())
    }

    /// Return the payload frames if the trailing MAC frame authenticates them.
    pub(crate) fn verify<'a>(&self, frames: &'a [Bytes]) -> Option<&'a [Bytes]> {
        let (mac, payload) = frames.split_last()?;
        self.mac(payload).verify_slice(mac).ok()?;
        Some(payload)
    }
}

impl fmt::Debug for CoordinatorAuthKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CoordinatorAuthKey(<redacted>)")
    }
}
