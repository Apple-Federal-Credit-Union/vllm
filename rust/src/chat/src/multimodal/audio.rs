// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Audio-modality preparation through `llm-multimodal`.

use std::sync::Arc;

use llm_multimodal::{AudioClip, Modality, PreprocessedEncoderInputs};
use vllm_engine_core_client::protocol::dtype::ModelDtype;

use super::timing::MM_STAGE_TARGET;
use super::{AudioModalitySupport, MultimodalModelInfo, PreparedMedia, item};
use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Forward-kwargs name of the primary audio encoder input.
pub(super) const AUDIO_PRIMARY_KEY: &str = "input_audio_features";

impl MultimodalModelInfo {
    /// Preprocess fetched audio clips as one batch and build per-item features.
    #[tracing::instrument(
        name = "mm_stage",
        target = MM_STAGE_TARGET,
        skip_all,
        fields(stage = "preprocess_audio")
    )]
    pub(super) async fn prepare_audios(
        &self,
        clips: Vec<Arc<AudioClip>>,
        uuids: Vec<Option<String>>,
    ) -> Result<PreparedMedia> {
        let support = self.audio.as_ref().ok_or_else(|| Error::UnsupportedModality {
            modality: Modality::Audio.to_string(),
        })?;
        let preprocessed = self.preprocess_audios(support, &clips).await?;
        let replacements = support.spec.prompt_replacements_for(&self.context, &preprocessed)?;
        if replacements.len() != clips.len() {
            bail_multimodal!(
                "number of audio prompt replacements {} does not match number of audio clips {}",
                replacements.len(),
                clips.len()
            );
        }

        let hashes = clips.iter().map(|clip| clip.hash.clone()).collect();
        let items = item::build_batched_items(
            &support.spec,
            preprocessed,
            hashes,
            uuids,
            ModelDtype::Float32,
        )?;

        Ok(PreparedMedia {
            modality: Modality::Audio,
            placeholder: support.placeholder.clone(),
            replacements,
            items,
        })
    }

    /// Run CPU-heavy audio preprocessing in a blocking task.
    async fn preprocess_audios(
        &self,
        support: &AudioModalitySupport,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs> {
        let processor = Arc::clone(&support.processor);
        let clips = clips.to_vec();
        tokio::task::spawn_blocking(move || Ok(processor.preprocess(&clips)?))
            .await
            .map_err(|error| multimodal!("audio preprocessing task failed: {error}"))?
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use llm_multimodal::{MediaContentPart, ModelSpecificValue, PreProcessorConfig};
    use ndarray::ArrayD;
    use vllm_engine_core_client::protocol::multimodal::{MmField, MmKwargValue};
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::super::{MultimodalModelContext, TokenizerResolver};
    use super::*;

    const AUDIO_PAD_ID: u32 = 151_676;
    const INKLING_AUDIO_MARKER_ID: u32 = 200_020;
    const INKLING_AUDIO_EMBED_ID: i32 = 200_053;
    const INKLING_AUDIO_SOFT_PLACEHOLDER: &str = "<|unused_200053|>";
    const INKLING_IMAGE_SOFT_PLACEHOLDER: &str = "<|unused_200054|>";

    fn inkling_info(decoder_dmodel: serde_json::Value) -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "inkling_mm_model",
            // Keep the vision tower enabled so audio gating alone decides
            // whether the model still resolves multimodal support.
            "vision_config": {"decoder_dmodel": 6144},
            "audio_config": {
                "decoder_dmodel": decoder_dmodel,
                "n_mel_bins": 80,
                "mel_vocab_size": 16,
                "dmel_min_value": -7.0,
                "dmel_max_value": 2.0
            }
        });
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<|content_image|>", 200_005)
            .with_regular_token("<|content_audio_input|>", INKLING_AUDIO_MARKER_ID)
            .with_regular_token(
                INKLING_AUDIO_SOFT_PLACEHOLDER,
                INKLING_AUDIO_EMBED_ID as u32,
            )
            .with_regular_token(INKLING_IMAGE_SOFT_PLACEHOLDER, 200_054);
        let context = MultimodalModelContext {
            model_id: "inkling-test".to_string(),
            model_type: Some("inkling_mm_model".to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(tokenizer)),
        };

        MultimodalModelInfo::from_loaded(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .expect("Inkling multimodal support")
    }

    fn qwen3_asr_info() -> MultimodalModelInfo {
        let context = MultimodalModelContext {
            model_id: "Qwen/Qwen3-ASR-1.7B".to_string(),
            model_type: Some("qwen3_asr".to_string()),
            config: serde_json::json!({"model_type": "qwen3_asr"}),
            tokenizer: TokenizerResolver(Arc::new(
                TestTokenizer::new().with_regular_token("<|audio_pad|>", AUDIO_PAD_ID),
            )),
        };

        MultimodalModelInfo::from_loaded(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .expect("Qwen3-ASR multimodal support")
    }

    fn wav_i16_mono(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let data_bytes = samples.len() as u32 * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn resolves_inkling_audio_from_model_spec() {
        let info = inkling_info(serde_json::json!(1024));
        let support = info.audio.as_ref().expect("audio support");

        // The spec renders the checkpoint's soft placeholder; the template's
        // `<|content_audio_input|>` marker rides in the structural prefix.
        assert_eq!(
            info.placeholder_token(Modality::Audio),
            Some(INKLING_AUDIO_SOFT_PLACEHOLDER)
        );
        assert_eq!(
            support.placeholder.marker_token_ids,
            vec![INKLING_AUDIO_EMBED_ID as u32]
        );
        assert_eq!(
            support.placeholder.embed_token_id,
            INKLING_AUDIO_EMBED_ID as u32
        );
        assert_eq!(support.spec.primary_key(), AUDIO_PRIMARY_KEY);
        assert!(matches!(
            &support.spec.field_layouts.encoder_input,
            llm_multimodal::FieldLayout::Flat { sizes_key }
                if sizes_key == "tokens_per_item"
        ));
        assert!(matches!(
            support.spec.field_layouts.model_specific.get("tokens_per_item"),
            Some(llm_multimodal::FieldLayout::Batched)
        ));
    }

    #[test]
    fn inkling_decoder_config_gates_audio_capability() {
        let info = inkling_info(serde_json::Value::Null);

        assert!(info.audio.is_none());
        assert_eq!(info.placeholder_token(Modality::Audio), None);
    }

    #[test]
    fn resolves_qwen_audio_from_model_spec() {
        let info = qwen3_asr_info();
        let support = info.audio.as_ref().expect("audio support");

        assert_eq!(
            info.placeholder_token(Modality::Audio),
            Some("<|audio_pad|>")
        );
        assert_eq!(support.placeholder.marker_token_ids, vec![AUDIO_PAD_ID]);
        assert_eq!(support.placeholder.embed_token_id, AUDIO_PAD_ID);
        assert!(matches!(
            support.spec.field_layouts.encoder_input,
            llm_multimodal::FieldLayout::Batched
        ));

        let preprocessed = PreprocessedEncoderInputs {
            encoder_input: ArrayD::zeros(vec![1, 128, 4]).into(),
            feature_token_counts: vec![1],
            item_sizes: vec![(128, 4)],
            model_specific: HashMap::from([
                (
                    "feature_attention_mask".to_string(),
                    ModelSpecificValue::int_2d(vec![1; 4], 1, 4),
                ),
                (
                    "audio_feature_lengths".to_string(),
                    ModelSpecificValue::int_1d(vec![4]),
                ),
            ]),
        };
        let item = item::build_batched_items(
            &support.spec,
            preprocessed,
            vec!["<hash>".to_string()],
            vec![None],
            ModelDtype::Float32,
        )
        .unwrap()
        .pop()
        .unwrap();

        assert!(matches!(
            &item.data[AUDIO_PRIMARY_KEY].field,
            MmField::Batched(_)
        ));
    }

    #[tokio::test]
    async fn audio_pipeline_attributes_stage_timings_to_request() {
        use tracing::Instrument as _;
        use tracing::instrument::WithSubscriber as _;
        use tracing_subscriber::layer::SubscriberExt as _;

        use crate::{mm_request_span, mm_timing_layer};

        let info = qwen3_asr_info();
        let (layer, timings) = mm_timing_layer();
        let parts = vec![MediaContentPart::AudioData {
            data: wav_i16_mono(16_000, &[0; 1_600]),
            mime_type: Some("audio/wav".to_string()),
            uuid: Some("audio-1".to_string()),
        }];
        async {
            info.prepare_multimodal(parts, &mut vec![AUDIO_PAD_ID], ModelDtype::Float32, None)
                .instrument(mm_request_span("request-1"))
                .await
                .unwrap();
        }
        .with_subscriber(tracing_subscriber::registry().with(layer))
        .await;

        let records = timings.stat();
        assert_eq!(records.len(), 1);
        let mut stages = records["request-1"].keys().collect::<Vec<_>>();
        stages.sort();
        expect_test::expect![[r#"
            [
                "media_fetch_secs",
                "preprocess_audio_secs",
                "preprocessor_total_secs",
                "prompt_expansion_secs",
            ]
        "#]]
        .assert_debug_eq(&stages);
    }

    #[tokio::test]
    async fn tracker_processor_and_lowering_preserve_audio_contract() {
        let info = qwen3_asr_info();
        let wav = wav_i16_mono(16_000, &[0; 1_600]);
        let expected_hash = llm_multimodal::hasher::hash_audio(&wav);
        let fetched = info
            .fetch_media(vec![MediaContentPart::AudioData {
                data: wav,
                mime_type: Some("audio/wav".to_string()),
                uuid: Some("audio-1".to_string()),
            }])
            .await
            .unwrap();

        let prepared = info.prepare_audios(fetched.audios, fetched.audio_uuids).await.unwrap();

        assert_eq!(prepared.replacements.len(), 1);
        assert!(
            prepared.replacements[0]
                .tokens
                .iter()
                .all(|token| *token == AUDIO_PAD_ID as i32)
        );
        let item = &prepared.items[0];
        assert_eq!(item.hash, expected_hash);
        assert_eq!(item.uuid.as_deref(), Some("audio-1"));

        let features = &item.data[AUDIO_PRIMARY_KEY];
        assert!(matches!(&features.field, MmField::Batched(_)));
        assert!(matches!(
            features.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "float32" && tensor.shape.first() == Some(&128)
        ));
        let lengths = &item.data["audio_feature_lengths"];
        assert!(matches!(&lengths.field, MmField::Batched(_)));
        assert!(matches!(
            lengths.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "int64" && tensor.shape.is_empty()
        ));
    }

    #[tokio::test]
    async fn inkling_tracker_and_processor_use_standard_audio_key() {
        let info = inkling_info(serde_json::json!(1024));
        let wav = wav_i16_mono(16_000, &[0; 1_600]);
        let expected_hash = llm_multimodal::hasher::hash_audio(&wav);
        let fetched = info
            .fetch_media(vec![MediaContentPart::AudioData {
                data: wav,
                mime_type: Some("audio/wav".to_string()),
                uuid: Some("audio-1".to_string()),
            }])
            .await
            .unwrap();

        let prepared = info.prepare_audios(fetched.audios, fetched.audio_uuids).await.unwrap();

        assert_eq!(prepared.replacements.len(), 1);
        // Every replacement token is the soft placeholder; the typed
        // `<|content_audio_input|>` marker is the structural prefix owned by
        // the chat template.
        assert!(
            prepared.replacements[0]
                .tokens
                .iter()
                .all(|token| *token == INKLING_AUDIO_EMBED_ID)
        );
        let item = &prepared.items[0];
        assert_eq!(item.hash, expected_hash);
        assert_eq!(item.uuid.as_deref(), Some("audio-1"));

        let features = &item.data[AUDIO_PRIMARY_KEY];
        assert!(matches!(&features.field, MmField::Flat(_)));
        assert!(matches!(
            features.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "float32" && tensor.shape.get(1) == Some(&80)
        ));
        let count = &item.data["tokens_per_item"];
        assert!(matches!(&count.field, MmField::Batched(_)));
        assert!(matches!(
            count.data.as_ref(),
            Some(MmKwargValue::Tensor(tensor))
                if tensor.dtype.as_str() == "int64" && tensor.shape.is_empty()
        ));
    }

    const MUSE_AUDIO_ID: u32 = 200_038;

    fn muse_spark_info() -> MultimodalModelInfo {
        let config = serde_json::json!({
            "model_type": "muse_spark",
            "audio_config": {"patch_sizes": [16, 6, 5, 4]},
        });
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<|audio|>", MUSE_AUDIO_ID)
            .with_regular_token("<|soft_audio_start|>", 201_463)
            .with_regular_token("<|soft_audio_end|>", 201_464);
        let context = MultimodalModelContext {
            model_id: "muse-spark-test".to_string(),
            model_type: Some("muse_spark".to_string()),
            config,
            tokenizer: TokenizerResolver(Arc::new(tokenizer)),
        };
        MultimodalModelInfo::from_loaded(
            context,
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            PreProcessorConfig::default(),
            HashMap::new(),
        )
        .unwrap()
        .expect("MuseSpark audio lane")
    }

    fn base64_encode(bytes: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        let mut index = 0;
        while index + 3 <= bytes.len() {
            let chunk = ((bytes[index] as u32) << 16)
                | ((bytes[index + 1] as u32) << 8)
                | bytes[index + 2] as u32;
            for shift in [18, 12, 6, 0] {
                out.push(TABLE[((chunk >> shift) & 0x3f) as usize] as char);
            }
            index += 3;
        }
        let rest = bytes.len() - index;
        if rest == 1 {
            let chunk = (bytes[index] as u32) << 16;
            out.push(TABLE[((chunk >> 18) & 0x3f) as usize] as char);
            out.push(TABLE[((chunk >> 12) & 0x3f) as usize] as char);
            out.push('=');
            out.push('=');
        } else if rest == 2 {
            let chunk = ((bytes[index] as u32) << 16) | ((bytes[index + 1] as u32) << 8);
            out.push(TABLE[((chunk >> 18) & 0x3f) as usize] as char);
            out.push(TABLE[((chunk >> 12) & 0x3f) as usize] as char);
            out.push(TABLE[((chunk >> 6) & 0x3f) as usize] as char);
            out.push('=');
        }
        out
    }

    fn input_audio_request(rates: &[u32]) -> crate::request::ChatRequest {
        let mut request = crate::request::ChatRequest::for_test();
        request.messages = vec![crate::request::ChatMessage::user(
            crate::request::ChatContent::Parts(
                rates
                    .iter()
                    .map(|&rate| {
                        crate::request::ChatContentPart::input_audio(
                            base64_encode(&wav_i16_mono(rate, &[0])),
                            Some("wav".to_string()),
                        )
                    })
                    .collect(),
            ),
        )];
        request
    }

    async fn prepare_input_audio(
        info: &MultimodalModelInfo,
        rates: &[u32],
        placeholder: u32,
    ) -> String {
        let request = input_audio_request(rates);
        let parts = super::super::extract_media_parts(&request, None).expect("input_audio parts");
        assert_eq!(parts.len(), rates.len());
        for part in &parts {
            match part {
                MediaContentPart::AudioUrl { url, .. } => {
                    assert!(url.starts_with("data:audio/wav;base64,"), "{url}");
                }
                other => panic!("input_audio must be fetched as an audio URL, got {other:?}"),
            }
        }
        let fetched = info.fetch_media(parts.clone()).await.expect("decode wav");
        assert_eq!(fetched.audios.len(), rates.len());
        for (clip, &rate) in fetched.audios.iter().zip(rates) {
            assert_eq!(clip.decoded().sample_rate, rate as usize);
            assert_eq!(clip.decoded().samples.len(), 1);
        }
        let mut tokens = vec![placeholder; rates.len()];
        match info.prepare_multimodal(parts, &mut tokens, ModelDtype::Float32, None).await {
            Ok(_) => "prepared audio without refusing the sample rate".to_string(),
            Err(error) => error.to_string(),
        }
    }

    fn assert_resample_refused(rate: u32, message: &str) {
        assert!(
            !message.contains("failed to allocate"),
            "{rate} Hz tried to allocate the kernel: {message}"
        );
        assert!(
            !message.contains("after resampling"),
            "{rate} Hz was resampled before the error: {message}"
        );
        let names_rate = message.contains(&rate.to_string());
        let names_bound = message.contains("kernel")
            && (message.contains("too large")
                || message.contains("practical")
                || message.contains("exceed")
                || message.contains("bound"));
        assert!(
            names_rate || names_bound,
            "expected {rate} Hz to be refused for its resample kernel, got: {message}"
        );
    }

    /// PCM WAV whose `fmt` rate is coprime with the model rate. One sample is
    /// enough: the sinc table depends on the reduced rates, not the length.
    #[tokio::test]
    async fn standard_wav_rates_still_prepare() {
        let info = muse_spark_info();
        for rate in [16_000, 44_100, 48_000] {
            let message = prepare_input_audio(&info, &[rate], MUSE_AUDIO_ID).await;
            assert!(
                message.contains("without refusing"),
                "{rate} Hz is a standard rate and must still preprocess: {message}"
            );
        }
    }

    /// 44 056 Hz shares only a factor of 8 with 24 kHz (5507:3000). The
    /// MuseSpark resampler asks for 16_593_000 kernel floats before reading
    /// the single sample. The same ratio against Qwen3 and Inkling's 16 kHz
    /// target is 11_082_000 floats. Both must be refused.
    #[tokio::test]
    async fn inband_coprime_wav_rate_is_refused_before_resample() {
        let hostile = 44_056;
        let muse = prepare_input_audio(&muse_spark_info(), &[24_000, hostile], MUSE_AUDIO_ID).await;
        assert_resample_refused(hostile, &muse);

        let qwen = prepare_input_audio(&qwen3_asr_info(), &[hostile], AUDIO_PAD_ID).await;
        assert_resample_refused(hostile, &qwen);

        let inkling = prepare_input_audio(
            &inkling_info(serde_json::json!(1024)),
            &[hostile],
            INKLING_AUDIO_EMBED_ID as u32,
        )
        .await;
        assert_resample_refused(hostile, &inkling);
    }

    /// 23 999 Hz and 240 007 Hz are coprime with 24 kHz. The kernel is
    /// `new_freq * kernel_len` floats: about 2.3 GiB and 23 GiB. The child
    /// process caps its address space so a vulnerable build fails this test
    /// on the allocation error instead of exhausting the machine.
    #[tokio::test]
    async fn unbounded_header_rate_is_refused_before_kernel_allocation() {
        if std::env::var_os("VLLM_RESAMPLE_PROBE").is_none() {
            let exe = std::env::current_exe().expect("test executable");
            let output = std::process::Command::new(exe)
                .args([
                    "--exact",
                    "multimodal::audio::tests::unbounded_header_rate_is_refused_before_kernel_allocation",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env("VLLM_RESAMPLE_PROBE", "1")
                .output()
                .expect("spawn probe");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stdout.contains("VLLM_RESAMPLE_PROBE_RAN")
                    || stderr.contains("VLLM_RESAMPLE_PROBE_RAN"),
                "probe did not run\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            assert!(
                output.status.success(),
                "probe status {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status
            );
            return;
        }
        eprintln!("VLLM_RESAMPLE_PROBE_RAN");

        let _guard = AddressSpaceGuard::cap_above_current(32 * 1024 * 1024);
        let info = muse_spark_info();
        for rate in [23_999, 240_007] {
            let message = prepare_input_audio(&info, &[rate], MUSE_AUDIO_ID).await;
            assert_resample_refused(rate, &message);
        }
    }

    #[repr(C)]
    struct RLimit {
        rlim_cur: u64,
        rlim_max: u64,
    }

    unsafe extern "C" {
        fn getrlimit(resource: i32, rlim: *mut RLimit) -> i32;
        fn setrlimit(resource: i32, rlim: *const RLimit) -> i32;
    }

    const RLIMIT_AS: i32 = 9;

    struct AddressSpaceGuard(RLimit);

    impl AddressSpaceGuard {
        fn cap_above_current(extra_bytes: u64) -> Self {
            let mut current = RLimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            let rc = unsafe { getrlimit(RLIMIT_AS, &mut current) };
            assert_eq!(rc, 0, "getrlimit(RLIMIT_AS) failed");
            let status = std::fs::read_to_string("/proc/self/status").expect("VmSize");
            let vm_kb = status
                .lines()
                .find_map(|line| line.strip_prefix("VmSize:"))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|kb| kb.parse::<u64>().ok())
                .expect("VmSize");
            let want = vm_kb.saturating_mul(1024).saturating_add(extra_bytes);
            let cur = if current.rlim_max == u64::MAX {
                want
            } else {
                want.min(current.rlim_max)
            };
            assert!(
                cur > vm_kb.saturating_mul(1024),
                "address-space cap {cur} is not above VmSize"
            );
            let tightened = RLimit {
                rlim_cur: cur,
                rlim_max: current.rlim_max,
            };
            let rc = unsafe { setrlimit(RLIMIT_AS, &tightened) };
            assert_eq!(rc, 0, "setrlimit(RLIMIT_AS) failed");
            Self(current)
        }
    }

    impl Drop for AddressSpaceGuard {
        fn drop(&mut self) {
            unsafe {
                setrlimit(RLIMIT_AS, &self.0);
            }
        }
    }
}
