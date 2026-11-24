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
    use vllm_engine_core_client::protocol::multimodal::{MmFeatureSpec, MmField, MmKwargValue};
    use vllm_tokenizer::test_utils::TestTokenizer;

    use crate::request::{ChatContentPart, ChatMessage, ChatRequest};

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

    const SHARED_AUDIO_UUID: &str = "doc-42";

    fn assert_audio_uuid_is_not_the_cache_key(
        first_tokens: &[u32],
        first: &MmFeatureSpec,
        second_tokens: &[u32],
        second: &MmFeatureSpec,
        repeated_first: &MmFeatureSpec,
    ) {
        assert_eq!(first_tokens, second_tokens);
        assert_eq!(first.mm_position.length, second.mm_position.length);
        assert_eq!(first.mm_position.is_embed, second.mm_position.is_embed);
        assert_ne!(first.mm_hash, second.mm_hash);
        assert_ne!(first.data, second.data);
        assert_ne!(first.identifier, SHARED_AUDIO_UUID);
        assert_ne!(second.identifier, SHARED_AUDIO_UUID);
        assert_ne!(first.identifier, second.identifier);
        assert_eq!(repeated_first.identifier, first.identifier);
    }

    async fn audio_feature_from_part(part: MediaContentPart) -> (Vec<u32>, MmFeatureSpec) {
        let info = qwen3_asr_info();
        let mut tokens = vec![AUDIO_PAD_ID];
        let features = info
            .prepare_multimodal(vec![part], &mut tokens, ModelDtype::Float32, None)
            .await
            .expect("audio feature");
        assert_eq!(features.len(), 1);
        (tokens, features.into_iter().next().unwrap())
    }

    fn audio_part(samples: &[i16], uuid: Option<&str>) -> MediaContentPart {
        MediaContentPart::AudioData {
            data: wav_i16_mono(16_000, samples),
            mime_type: Some("audio/wav".to_string()),
            uuid: uuid.map(str::to_string),
        }
    }

    #[tokio::test]
    async fn audio_client_uuid_is_not_the_encoder_or_prefix_cache_key() {
        let tone = [0i16; 1_600];
        let mut other = tone;
        other[0] = 8_000;
        let (first_tokens, first) =
            audio_feature_from_part(audio_part(&tone, Some(SHARED_AUDIO_UUID))).await;
        let (second_tokens, second) =
            audio_feature_from_part(audio_part(&other, Some(SHARED_AUDIO_UUID))).await;
        let (_, repeated) =
            audio_feature_from_part(audio_part(&tone, Some(SHARED_AUDIO_UUID))).await;
        assert_audio_uuid_is_not_the_cache_key(
            &first_tokens,
            &first,
            &second_tokens,
            &second,
            &repeated,
        );
    }

    fn audio_b64(bytes: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        let mut index = 0;
        while index + 3 <= bytes.len() {
            let packed = ((bytes[index] as u32) << 16)
                | ((bytes[index + 1] as u32) << 8)
                | bytes[index + 2] as u32;
            out.push(TABLE[((packed >> 18) & 63) as usize] as char);
            out.push(TABLE[((packed >> 12) & 63) as usize] as char);
            out.push(TABLE[((packed >> 6) & 63) as usize] as char);
            out.push(TABLE[(packed & 63) as usize] as char);
            index += 3;
        }
        if index < bytes.len() {
            let b0 = bytes[index];
            let b1 = if index + 1 < bytes.len() {
                bytes[index + 1]
            } else {
                0
            };
            let packed = ((b0 as u32) << 16) | ((b1 as u32) << 8);
            out.push(TABLE[((packed >> 18) & 63) as usize] as char);
            out.push(TABLE[((packed >> 12) & 63) as usize] as char);
            if index + 1 < bytes.len() {
                out.push(TABLE[((packed >> 6) & 63) as usize] as char);
                out.push('=');
            } else {
                out.push('=');
                out.push('=');
            }
        }
        out
    }

    async fn assert_chat_audio_parts_do_not_share_uuid(
        parts_for: fn(&[u8]) -> Vec<MediaContentPart>,
    ) {
        let tone = wav_i16_mono(16_000, &[0i16; 1_600]);
        let mut samples = [0i16; 1_600];
        samples[0] = 8_000;
        let other = wav_i16_mono(16_000, &samples);
        let (first_tokens, first) = audio_feature_from_part(parts_for(&tone).remove(0)).await;
        let (second_tokens, second) = audio_feature_from_part(parts_for(&other).remove(0)).await;
        let (_, repeated) = audio_feature_from_part(parts_for(&tone).remove(0)).await;
        assert_audio_uuid_is_not_the_cache_key(
            &first_tokens,
            &first,
            &second_tokens,
            &second,
            &repeated,
        );
    }

    fn input_audio_parts(bytes: &[u8]) -> Vec<MediaContentPart> {
        let request = ChatRequest {
            messages: vec![ChatMessage::user(vec![ChatContentPart::InputAudio {
                data: audio_b64(bytes),
                format: Some("wav".to_string()),
                uuid: Some(SHARED_AUDIO_UUID.to_string()),
            }])],
            ..ChatRequest::for_test()
        };
        super::super::extract_media_parts(&request, None).expect("input_audio part")
    }

    fn audio_url_parts(bytes: &[u8]) -> Vec<MediaContentPart> {
        let request = ChatRequest {
            messages: vec![ChatMessage::user(vec![ChatContentPart::AudioUrl {
                audio_url: format!("data:audio/wav;base64,{}", audio_b64(bytes)),
                uuid: Some(SHARED_AUDIO_UUID.to_string()),
            }])],
            ..ChatRequest::for_test()
        };
        super::super::extract_media_parts(&request, None).expect("audio_url part")
    }

    #[tokio::test]
    async fn input_audio_uuid_is_not_the_encoder_or_prefix_cache_key() {
        assert_chat_audio_parts_do_not_share_uuid(input_audio_parts).await;
    }

    #[tokio::test]
    async fn audio_url_uuid_is_not_the_encoder_or_prefix_cache_key() {
        assert_chat_audio_parts_do_not_share_uuid(audio_url_parts).await;
    }
}
