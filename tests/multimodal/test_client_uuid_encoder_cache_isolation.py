# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Client media UUIDs must not select another request's encoder output.

These tests fail while a client-supplied ``uuid`` is the encoder-cache
identity. They pass once distinct media cannot resolve to the same cached
embeddings, including when the embedding counts match.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from PIL import Image

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.multimodal.cache import (
    LruKeyReplicatedReceiverCache,
    LruKeyReplicatedSenderCache,
    MultiModalProcessorOnlyCache,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.parse import MultiModalDataParser, parse_mm_uuids
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test

EMBED_LEN = 4


@dataclass
class _ReqIdBatch:
    """The request-id list `GPUModelRunner._gather_mm_embeddings` reads."""

    req_ids: list[str]


MediaKind = str
_KINDS = (
    "image",
    "audio",
    "video",
    "vision_chunk",
    "image_embeds",
    "audio_embeds",
    "video_embeds",
)


@pytest.fixture(scope="module")
def scheduler() -> Scheduler:
    return create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=2048,
        max_num_batched_tokens=2048,
    )


def _identifier(mm_hash: str) -> str:
    """Run the real identifier helper with no LoRA and no caller scope."""
    processor = InputProcessor.__new__(InputProcessor)
    processor.lora_config = None
    return processor._get_mm_identifier(mm_hash, None)


def _pixels(seed: int) -> torch.Tensor:
    return torch.full((EMBED_LEN, 3), float(seed), dtype=torch.float32)


def _kwargs_item(seed: int) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            "pixels": MultiModalFieldElem(
                data=_pixels(seed),
                field=MultiModalSharedField(batch_size=1),
            )
        }
    )


def _payloads(kind: MediaKind) -> tuple[dict, dict, str]:
    """Two different one-item media dicts and the modality key they hash under."""
    if kind == "image":
        return (
            {"image": [Image.new("RGB", (8, 8), (255, 0, 0))]},
            {"image": [Image.new("RGB", (8, 8), (0, 0, 255))]},
            "image",
        )
    if kind == "audio":
        return (
            {"audio": [np.arange(16, dtype=np.float32)]},
            {"audio": [np.arange(16, dtype=np.float32) + 10]},
            "audio",
        )
    if kind == "video":
        return (
            {"video": [np.zeros((2, 4, 4, 3), dtype=np.uint8)]},
            {"video": [np.ones((2, 4, 4, 3), dtype=np.uint8)]},
            "video",
        )
    if kind == "vision_chunk":
        red = Image.new("RGB", (4, 4), (255, 0, 0))
        blue = Image.new("RGB", (4, 4), (0, 0, 255))
        return (
            {"vision_chunk": [{"type": "image", "image": red}]},
            {"vision_chunk": [{"type": "image", "image": blue}]},
            "vision_chunk",
        )
    if kind == "image_embeds":
        return (
            {"image": [torch.ones(EMBED_LEN, 3)]},
            {"image": [torch.zeros(EMBED_LEN, 3)]},
            "image",
        )
    if kind == "audio_embeds":
        return (
            {"audio": [torch.ones(EMBED_LEN, 3)]},
            {"audio": [torch.zeros(EMBED_LEN, 3)]},
            "audio",
        )
    if kind == "video_embeds":
        return (
            {"video": [torch.ones(EMBED_LEN, 3)]},
            {"video": [torch.zeros(EMBED_LEN, 3)]},
            "video",
        )
    raise AssertionError(kind)


def _factors(
    modality: str, enabled: bool
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if not enabled:
        return {}, {}
    if modality == "image":
        return {"images_kwargs": {"do_rescale": True}}, {"image": {"size": 224}}
    if modality == "audio":
        return {"audio_kwargs": {"padding": True}}, {"audio": {"sampling_rate": 16000}}
    if modality == "video":
        return {"videos_kwargs": {"do_resize": True}}, {"video": {"num_frames": 8}}
    return {"mm_processor_kwargs": {"do_resize": True}}, {modality: {"size": 224}}


def _hashes(
    mm_data: dict,
    uuids: list[str | None],
    *,
    model_id: str,
    algorithm: str,
    processor_kwargs: dict[str, object],
    media_io_kwargs: dict[str, dict[str, object]],
) -> list[str]:
    modality = next(iter(mm_data))
    items = MultiModalDataParser().parse_mm_data(mm_data)
    assert len(items[modality]) == len(uuids)
    inputs = ProcessorInputs(
        prompt=[0],
        mm_data_items=items,
        mm_uuid_items=parse_mm_uuids({modality: uuids}),
        hf_processor_mm_kwargs=processor_kwargs,
        media_io_kwargs=media_io_kwargs,
    )
    return inputs.get_mm_hashes(model_id, algorithm)[modality]


def _feature(
    identifier: str, item: MultiModalKwargsItem, modality: str, offset: int
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=item,
        modality=modality,
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=EMBED_LEN),
        mm_hash=identifier,
    )


def _request_with_features(request_id: str, features: list[MultiModalFeatureSpec]):
    num_tokens = EMBED_LEN * len(features)
    request = create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        mm_hashes_list=[[feature.identifier for feature in features]],
        mm_positions=[[feature.mm_position for feature in features]],
        req_ids=[request_id],
        max_tokens=1,
    )[0]
    request.mm_features = features
    return request


def _gather_v2(
    req_id: str, features: list[MultiModalFeatureSpec], cache: EncoderCache
) -> list[torch.Tensor]:
    runner = EncoderRunner.__new__(EncoderRunner)
    runner.is_realtime = False
    runner.encoder_cache = cache
    cache.add_request(req_id, features)
    num_tokens = EMBED_LEN * len(features)
    embeds, _ = runner.gather_mm_embeddings(
        [req_id],
        num_tokens,
        np.array([num_tokens], dtype=np.int32),
        np.array([0, num_tokens], dtype=np.int32),
        np.array([num_tokens], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )
    return embeds


def _gather_v1(
    req_id: str, features: list[MultiModalFeatureSpec], outputs: dict[str, torch.Tensor]
) -> list[torch.Tensor]:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_multimodal_pruning_enabled = False
    runner.encoder_cache = outputs
    runner.requests = {
        req_id: CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[0] * (EMBED_LEN * len(features)),
            mm_features=features,
            sampling_params=None,
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
    }
    runner.input_batch = _ReqIdBatch(req_ids=[req_id])
    scheduler_output = SchedulerOutput.make_empty()
    num_tokens = EMBED_LEN * len(features)
    scheduler_output.num_scheduled_tokens = {req_id: num_tokens}
    scheduler_output.total_num_scheduled_tokens = num_tokens
    embeds, _ = runner._gather_mm_embeddings(scheduler_output)
    return embeds


def _assert_embeddings_are_this_request(
    embeds: list[torch.Tensor], expected: list[torch.Tensor]
) -> None:
    assert len(embeds) == len(expected)
    for got, want in zip(embeds, expected):
        assert torch.equal(got, want), "encoder output is another request's media"


def _processor_caches_return_this_item(
    model_config,
    victim_hash: str,
    victim_item: MultiModalKwargsItem,
    attacker_hash: str,
    attacker_item: MultiModalKwargsItem,
) -> None:
    only = MultiModalProcessorOnlyCache(model_config)
    only.get_and_update_item((victim_item, ()), victim_hash)
    assert only.is_cached_item(victim_hash)
    returned, _ = only.get_and_update_item((attacker_item, ()), attacker_hash)
    assert torch.equal(returned["pixels"].data, attacker_item["pixels"].data)

    sender = LruKeyReplicatedSenderCache(model_config)
    receiver = LruKeyReplicatedReceiverCache(model_config)
    sender.get_and_update_item((victim_item, ()), victim_hash)
    receiver.get_and_update_item(victim_item, victim_hash)
    if sender.is_cached_item(attacker_hash):
        received = receiver.get_and_update_item(None, attacker_hash)
    else:
        received = receiver.get_and_update_item(attacker_item, attacker_hash)
    assert torch.equal(received["pixels"].data, attacker_item["pixels"].data)


def _runner_encodes_this_item(
    identifier: str,
    cached_identifier: str,
    cached_output: torch.Tensor,
    item: MultiModalKwargsItem,
    modality: str,
) -> None:
    cache = EncoderCache()
    cache.encoder_outputs[cached_identifier] = cached_output
    feature = _feature(identifier, item, modality, 0)
    cache.add_request("attacker", [feature])
    runner = EncoderRunner.__new__(EncoderRunner)
    runner.encoder_cache = cache
    _hashes_out, kwargs = runner.prepare_mm_inputs({"attacker": [0]})
    assert kwargs, "encoder runner skipped encoding this request's media"
    assert torch.equal(kwargs[0][1]["pixels"].data, item["pixels"].data)


def _schedule_and_gather(
    scheduler: Scheduler,
    *,
    owner_id: str,
    other_id: str,
    owner_features: list[MultiModalFeatureSpec],
    other_features: list[MultiModalFeatureSpec],
    owner_outputs: list[torch.Tensor],
    other_outputs: list[torch.Tensor],
) -> None:
    owner = _request_with_features(owner_id, owner_features)
    scheduler.add_request(owner)
    owner_sched = scheduler.schedule()
    assert owner_id in owner_sched.scheduled_encoder_inputs, (
        "owner media was not encoded, so the cache was not populated"
    )

    cache = EncoderCache()
    v1_outputs: dict[str, torch.Tensor] = {}
    for index in owner_sched.scheduled_encoder_inputs[owner_id]:
        identifier = owner_features[index].identifier
        cache.encoder_outputs[identifier] = owner_outputs[index]
        v1_outputs[identifier] = owner_outputs[index]

    other = _request_with_features(other_id, other_features)
    scheduler.add_request(other)
    other_sched = scheduler.schedule()
    try:
        if other_id in scheduler.encoder_cache_mismatch_reqs:
            assert other_id not in other_sched.num_scheduled_tokens
            return
        assert other_id in other_sched.num_scheduled_tokens, (
            "request was neither encoded nor rejected"
        )
        scheduled = set(other_sched.scheduled_encoder_inputs.get(other_id, ()))
        for index, feature in enumerate(other_features):
            if index in scheduled:
                cache.encoder_outputs[feature.identifier] = other_outputs[index]
                v1_outputs[feature.identifier] = other_outputs[index]
        _assert_embeddings_are_this_request(
            _gather_v2(other_id, other_features, cache), other_outputs
        )
        _assert_embeddings_are_this_request(
            _gather_v1(other_id, other_features, v1_outputs), other_outputs
        )
    finally:
        scheduler.finish_requests([owner_id, other_id], RequestStatus.FINISHED_ABORTED)


def _model_hash_args(scheduler: Scheduler):
    model_config = scheduler.vllm_config.model_config
    mm_config = model_config.get_multimodal_config()
    assert mm_config.mm_processor_cache_gb > 0
    return model_config, model_config.model, mm_config.mm_hasher_algorithm


@pytest.mark.parametrize("kind", _KINDS)
def test_shared_uuid_does_not_reuse_another_requests_embeddings(
    kind: MediaKind, scheduler: Scheduler
) -> None:
    """Different media that share a client uuid must not share encoder outputs.

    Covers image, audio, video, vision chunks, and precomputed embeds, with and
    without processor settings, and with either party populating the uuid first.
    """
    model_config, model_id, algorithm = _model_hash_args(scheduler)
    first_data, second_data, modality = _payloads(kind)
    for use_factors in (False, True):
        processor_kwargs, media_io_kwargs = _factors(modality, use_factors)
        for attacker_cached_first in (False, True):
            label = (
                f"{kind} factors={use_factors} attacker_first={attacker_cached_first}"
            )
            client_uuid = f"shared-{label}"
            first_hash = _identifier(
                _hashes(
                    first_data,
                    [client_uuid],
                    model_id=model_id,
                    algorithm=algorithm,
                    processor_kwargs=processor_kwargs,
                    media_io_kwargs=media_io_kwargs,
                )[0]
            )
            second_hash = _identifier(
                _hashes(
                    second_data,
                    [client_uuid],
                    model_id=model_id,
                    algorithm=algorithm,
                    processor_kwargs=processor_kwargs,
                    media_io_kwargs=media_io_kwargs,
                )[0]
            )
            first_item = _kwargs_item(1)
            second_item = _kwargs_item(2)
            _processor_caches_return_this_item(
                model_config, first_hash, first_item, second_hash, second_item
            )
            _runner_encodes_this_item(
                second_hash,
                first_hash,
                first_item["pixels"].data,
                second_item,
                modality,
            )
            if attacker_cached_first:
                owner_hash, other_hash = second_hash, first_hash
                owner_item, other_item = second_item, first_item
                owner_id, other_id = f"{label}-a", f"{label}-b"
            else:
                owner_hash, other_hash = first_hash, second_hash
                owner_item, other_item = first_item, second_item
                owner_id, other_id = f"{label}-b", f"{label}-a"
            _schedule_and_gather(
                scheduler,
                owner_id=owner_id,
                other_id=other_id,
                owner_features=[_feature(owner_hash, owner_item, modality, 0)],
                other_features=[_feature(other_hash, other_item, modality, 0)],
                owner_outputs=[owner_item["pixels"].data],
                other_outputs=[other_item["pixels"].data],
            )


@pytest.mark.parametrize("kind", ("image", "audio", "video"))
def test_one_request_does_not_alias_two_media_with_one_uuid(
    kind: MediaKind, scheduler: Scheduler
) -> None:
    """One prompt with two different items and one uuid keeps both embeddings."""
    _model_config, model_id, algorithm = _model_hash_args(scheduler)
    left_data, right_data, modality = _payloads(kind)
    combined = {modality: [left_data[modality][0], right_data[modality][0]]}
    client_uuid = f"shared-within-{kind}"
    for use_factors in (False, True):
        processor_kwargs, media_io_kwargs = _factors(modality, use_factors)
        raw_hashes = _hashes(
            combined,
            [client_uuid, client_uuid],
            model_id=model_id,
            algorithm=algorithm,
            processor_kwargs=processor_kwargs,
            media_io_kwargs=media_io_kwargs,
        )
        identifiers = [_identifier(item_hash) for item_hash in raw_hashes]
        left_item, right_item = _kwargs_item(3), _kwargs_item(4)
        features = [
            _feature(identifiers[0], left_item, modality, 0),
            _feature(identifiers[1], right_item, modality, EMBED_LEN),
        ]
        request_id = f"within-{kind}-{use_factors}"
        request = _request_with_features(request_id, features)
        scheduler.add_request(request)
        scheduled = scheduler.schedule()
        try:
            if request_id in scheduler.encoder_cache_mismatch_reqs:
                assert request_id not in scheduled.num_scheduled_tokens
                continue
            assert request_id in scheduled.num_scheduled_tokens
            encoded = set(scheduled.scheduled_encoder_inputs.get(request_id, ()))
            assert encoded == {0, 1}, (
                "scheduler reused one uuid for two different media items"
            )
            cache = EncoderCache()
            v1_outputs: dict[str, torch.Tensor] = {}
            expected = [left_item["pixels"].data, right_item["pixels"].data]
            for index in (0, 1):
                cache.encoder_outputs[features[index].identifier] = expected[index]
                v1_outputs[features[index].identifier] = expected[index]
            _assert_embeddings_are_this_request(
                _gather_v2(request_id, features, cache), expected
            )
            _assert_embeddings_are_this_request(
                _gather_v1(request_id, features, v1_outputs), expected
            )
        finally:
            scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)


def test_identical_media_and_uuid_still_share_one_encoder_output(
    scheduler: Scheduler,
) -> None:
    """The same bytes and uuid remain one cache entry."""
    _model_config, model_id, algorithm = _model_hash_args(scheduler)
    media, _other, modality = _payloads("image")
    client_uuid = "same-bytes-same-uuid"
    item_hash = _identifier(
        _hashes(
            media,
            [client_uuid],
            model_id=model_id,
            algorithm=algorithm,
            processor_kwargs={},
            media_io_kwargs={},
        )[0]
    )
    again = _identifier(
        _hashes(
            media,
            [client_uuid],
            model_id=model_id,
            algorithm=algorithm,
            processor_kwargs={},
            media_io_kwargs={},
        )[0]
    )
    assert item_hash == again
    item = _kwargs_item(7)
    feature = _feature(item_hash, item, modality, 0)
    _schedule_and_gather(
        scheduler,
        owner_id="same-owner",
        other_id="same-other",
        owner_features=[feature],
        other_features=[_feature(again, item, modality, 0)],
        owner_outputs=[item["pixels"].data],
        other_outputs=[item["pixels"].data],
    )


def test_chat_image_uuid_reaches_the_encoder_cache_key(scheduler: Scheduler) -> None:
    """A chat image part's uuid is the cache identity unless the bytes are in it."""
    from vllm.renderers.chat_utils import parse_chat_messages

    model_config, model_id, algorithm = _model_hash_args(scheduler)
    red = Image.new("RGB", (8, 8), (255, 0, 0))
    blue = Image.new("RGB", (8, 8), (0, 255, 0))
    client_uuid = "chat-image-uuid"

    def _from_chat(image: Image.Image) -> str:
        _conversation, mm_data, mm_uuids = parse_chat_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": image, "uuid": client_uuid},
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
            model_config,
            content_format="string",
        )
        assert mm_uuids is not None
        assert mm_uuids["image"] == [client_uuid]
        assert mm_data is not None
        return _identifier(
            _hashes(
                mm_data,
                mm_uuids["image"],
                model_id=model_id,
                algorithm=algorithm,
                processor_kwargs={},
                media_io_kwargs={},
            )[0]
        )

    red_hash = _from_chat(red)
    blue_hash = _from_chat(blue)
    red_item, blue_item = _kwargs_item(8), _kwargs_item(9)
    _processor_caches_return_this_item(
        model_config, red_hash, red_item, blue_hash, blue_item
    )
    _schedule_and_gather(
        scheduler,
        owner_id="chat-red",
        other_id="chat-blue",
        owner_features=[_feature(red_hash, red_item, "image", 0)],
        other_features=[_feature(blue_hash, blue_item, "image", 0)],
        owner_outputs=[red_item["pixels"].data],
        other_outputs=[blue_item["pixels"].data],
    )
