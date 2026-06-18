from __future__ import annotations

import hashlib
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semceb.algorithms.cardinality_estimate import (
    CardinalityEstimate,
    CardinalityEstimateKind,
)
from semceb.algorithms.interface import AlgorithmInterface
from semceb.queries.query_specification import QuerySpecification
from semceb.queries.template_parser import QueryTemplatePartType
from semceb.utils.console import console


CACHE_SCHEMA_VERSION = 3
ANSWER_PREFIX = "Answer: "
SUPPORTED_SOURCE_COLUMNS = (
    "product_title",
    "review_title",
    "review_text",
    "description_json",
    "details_json",
    "features_json",
)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return sanitized.strip("._") or "value"


def _cache_size_bytes(past_key_values: tuple[tuple[Any, ...], ...]) -> int:
    size = 0
    for layer in past_key_values:
        for tensor in layer:
            size += int(tensor.numel() * tensor.element_size())
    return size


def _tensor_size_bytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def _to_legacy_cache_on_cpu(past_key_values: Any) -> tuple[tuple[Any, ...], ...]:
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    return tuple(
        tuple(tensor.detach().cpu().contiguous() for tensor in layer)
        for layer in past_key_values
    )


@dataclass
class RowPrefixCache:
    prefix_length: int
    prefix_input_ids: Any
    past_key_values: tuple[tuple[Any, ...], ...]


@dataclass
class PreparedColumnState:
    dataset_name: str
    source_column: str
    estimator: Any
    threshold_estimator: "TextPreloadedKVThresholdEstimator"
    total_rows: int
    valid_rows: int
    memory_bytes: int


class LocalHFTextKVBackend:
    def __init__(self, model_name: str, device: str, max_new_tokens: int):
        self.model_name = model_name
        self.device_label = device
        self.max_new_tokens = max_new_tokens
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None
        self._chat_template_parts: tuple[str, str] | None = None
        self._binary_token_ids: tuple[int, int] | None = None

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def device(self):
        self._ensure_loaded()
        return self._device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
        AutoTokenizer = getattr(transformers, "AutoTokenizer")

        requested_device = self.resolve_device_label(torch, self.device_label)

        self._device = torch.device(requested_device)
        self._torch = torch

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        torch_dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(self._device)
        self._model.eval()

        if getattr(self._model.config, "pad_token_id", None) is None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id

    @staticmethod
    def resolve_device_label(torch: Any, configured_device: str) -> str:
        requested_device = configured_device.strip().lower()

        if requested_device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
            return "cpu"

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"

        if requested_device == "mps":
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                return "cpu"

        return requested_device

    def build_prefix_cache(self, prefix_text: str) -> RowPrefixCache:
        self._ensure_loaded()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        encoded = self._tokenizer(
            prefix_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with self._torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

        return RowPrefixCache(
            prefix_length=int(attention_mask.shape[1]),
            prefix_input_ids=input_ids.detach().cpu().contiguous(),
            past_key_values=_to_legacy_cache_on_cpu(outputs.past_key_values),
        )

    def build_row_prefix(self, row_text: str) -> str:
        chat_prefix, _ = self._get_chat_template_parts()
        normalized_row_text = _normalize_whitespace(row_text)
        return (
            f"{chat_prefix}"
            "Answer the following question based on the row text with '1' or '0'. "
            "Do not add any other comments.\n\n"
            f"Row text:\n{normalized_row_text}\n\n"
        )

    def build_runtime_suffix(self, predicate_text: str) -> str:
        _, chat_suffix = self._get_chat_template_parts()
        normalized_predicate = _normalize_whitespace(predicate_text)
        return (
            "Question: Does the row text satisfy this predicate?\n"
            f"Predicate:\n{normalized_predicate}\n"
            f"{chat_suffix}"
            f"{ANSWER_PREFIX}"
        )

    def evaluate_binary(
        self,
        runtime_question: str,
        cache: RowPrefixCache,
    ) -> tuple[str, dict[str, int]]:
        responses, stats = self.evaluate_binary_batch(runtime_question, [cache])
        return responses[0], stats

    def evaluate_binary_batch(
        self,
        runtime_question: str,
        caches: list[RowPrefixCache],
    ) -> tuple[list[str], dict[str, int]]:
        self._ensure_loaded()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        if not caches:
            return [], {"usd": 0, "llm_calls": 0, "model_batches": 0, "tokens": 0}

        encoded = self._tokenizer(
            runtime_question,
            return_tensors="pt",
            add_special_tokens=False,
        )
        suffix_input_ids = encoded["input_ids"].to(self._device)
        suffix_attention_mask = encoded["attention_mask"].to(self._device)

        dynamic_cache, prefix_attention_mask = self._build_batched_cache(caches)
        suffix_input_ids = suffix_input_ids.expand(len(caches), -1)
        suffix_attention_mask = suffix_attention_mask.expand(len(caches), -1)
        full_attention_mask = self._torch.cat(
            [prefix_attention_mask, suffix_attention_mask],
            dim=1,
        )

        with self._torch.no_grad():
            outputs = self._model(
                input_ids=suffix_input_ids,
                attention_mask=full_attention_mask,
                past_key_values=dynamic_cache,
                use_cache=False,
                return_dict=True,
            )

        zero_token_id, one_token_id = self._get_binary_token_ids()
        next_token_logits = outputs.logits[:, -1, :]
        zero_logits = next_token_logits[:, zero_token_id]
        one_logits = next_token_logits[:, one_token_id]
        responses = [
            "1" if is_positive else "0"
            for is_positive in (one_logits > zero_logits).detach().cpu().tolist()
        ]

        stats = {
            "usd": 0,
            "llm_calls": len(caches),
            "model_batches": 1,
            "tokens": int(suffix_input_ids.numel() + len(caches)),
        }
        return responses, stats

    def _build_batched_cache(self, caches: list[RowPrefixCache]) -> tuple[Any, Any]:
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._device is not None

        max_prefix_length = max(cache.prefix_length for cache in caches)
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        prefix_attention_masks = []
        batched_layers = []

        for layer_entries in zip(*(cache.past_key_values for cache in caches)):
            padded_keys = []
            padded_values = []
            for key_states, value_states in layer_entries:
                key_states = key_states.to(self._device)
                value_states = value_states.to(self._device)
                pad_len = max_prefix_length - int(key_states.shape[2])
                if pad_len > 0:
                    key_states = self._torch.nn.functional.pad(
                        key_states,
                        (0, 0, pad_len, 0),
                    )
                    value_states = self._torch.nn.functional.pad(
                        value_states,
                        (0, 0, pad_len, 0),
                    )
                else:
                    key_states = key_states.clone()
                    value_states = value_states.clone()
                padded_keys.append(key_states.contiguous())
                padded_values.append(value_states.contiguous())

            batched_layers.append(
                (
                    self._torch.cat(padded_keys, dim=0),
                    self._torch.cat(padded_values, dim=0),
                )
            )

        for cache in caches:
            pad_len = max_prefix_length - cache.prefix_length
            padding_mask = self._torch.zeros(
                (1, pad_len),
                dtype=self._torch.long,
                device=self._device,
            )
            prefix_mask = self._torch.ones(
                (1, cache.prefix_length),
                dtype=self._torch.long,
                device=self._device,
            )
            prefix_attention_masks.append(
                self._torch.cat([padding_mask, prefix_mask], dim=1)
            )

        transformers_cache_utils = importlib.import_module("transformers.cache_utils")
        DynamicCache = getattr(transformers_cache_utils, "DynamicCache")
        dynamic_cache = DynamicCache()
        for layer_idx, (key_states, value_states) in enumerate(batched_layers):
            dynamic_cache.update(key_states, value_states, layer_idx)

        return dynamic_cache, self._torch.cat(prefix_attention_masks, dim=0)

    def _get_chat_template_parts(self) -> tuple[str, str]:
        self._ensure_loaded()
        assert self._tokenizer is not None

        if self._chat_template_parts is not None:
            return self._chat_template_parts

        dummy = "SEMCEB_CHAT_TEMPLATE_PLACEHOLDER"
        if (
            not hasattr(self._tokenizer, "apply_chat_template")
            or getattr(self._tokenizer, "chat_template", None) is None
        ):
            self._chat_template_parts = ("", "\n")
            return self._chat_template_parts

        prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": dummy}],
            add_generation_prompt=True,
            tokenize=False,
        )
        if dummy not in prompt:
            self._chat_template_parts = ("", "\n")
            return self._chat_template_parts

        self._chat_template_parts = prompt.split(dummy, 1)
        return self._chat_template_parts

    def _get_binary_token_ids(self) -> tuple[int, int]:
        assert self._tokenizer is not None

        if self._binary_token_ids is not None:
            return self._binary_token_ids

        answer_prefix_ids = self._tokenizer.encode(
            ANSWER_PREFIX,
            add_special_tokens=False,
        )
        zero_with_prefix_ids = self._tokenizer.encode(
            f"{ANSWER_PREFIX}0",
            add_special_tokens=False,
        )
        one_with_prefix_ids = self._tokenizer.encode(
            f"{ANSWER_PREFIX}1",
            add_special_tokens=False,
        )
        if (
            zero_with_prefix_ids[: len(answer_prefix_ids)] == answer_prefix_ids
            and one_with_prefix_ids[: len(answer_prefix_ids)] == answer_prefix_ids
        ):
            zero_ids = zero_with_prefix_ids[len(answer_prefix_ids) :]
            one_ids = one_with_prefix_ids[len(answer_prefix_ids) :]
        else:
            zero_ids = self._tokenizer.encode("0", add_special_tokens=False)
            one_ids = self._tokenizer.encode("1", add_special_tokens=False)

        if len(zero_ids) != 1 or len(one_ids) != 1:
            raise RuntimeError(
                "SemanticHistogram binary logit scoring requires '0' and '1' to "
                "tokenize to one token each."
            )

        self._binary_token_ids = (int(zero_ids[0]), int(one_ids[0]))
        return self._binary_token_ids

class TextPreloadedKVThresholdEstimator:
    def __init__(
        self,
        dataset_name: str,
        source_column: str,
        row_texts: list[str],
        backend: LocalHFTextKVBackend,
        cache_dir: Path,
        hf_model_name: str,
        num_kv_caches: int,
        eval_batch_size: int,
        seed: int,
    ):
        self.dataset_name = dataset_name
        self.source_column = source_column
        self.row_texts = row_texts
        self.backend = backend
        self.cache_dir = cache_dir
        self.hf_model_name = hf_model_name
        self.num_kv_caches = num_kv_caches
        self.eval_batch_size = eval_batch_size
        self.seed = seed

        self.image_embeddings = None
        self.row_prefix_caches: list[RowPrefixCache] = []
        self.selected_indices: list[int] = []
        self.cost_stats = {"usd": 0, "llm_calls": 0, "model_batches": 0, "tokens": 0}
        self.cache_bytes = 0

    def name(self):
        return type(self).__name__

    def is_deterministic(self) -> bool:
        return True

    def fit(
        self,
        images: list[Path],
        image_embeddings: Any,
        n_components: int,
        seed: int,
    ):
        del images
        del seed

        self.image_embeddings = image_embeddings
        artifact_path = self._cache_artifact_path(n_components=n_components)
        console.print(
            "[cyan]SemanticHistogram[/cyan] cache prep: checking for cached KV state at "
            f"[bold]{artifact_path}[/bold]."
        )
        artifact = self._load_cache_artifact(n_components=n_components)
        if artifact is not None:
            console.print(
                "[cyan]SemanticHistogram[/cyan] cache prep: reusing cached KV state."
            )
            self.selected_indices = artifact["selected_indices"]
            self.row_prefix_caches = [
                RowPrefixCache(
                    prefix_length=entry["prefix_length"],
                    prefix_input_ids=entry["prefix_input_ids"],
                    past_key_values=entry["past_key_values"],
                )
                for entry in artifact["row_prefix_caches"]
            ]
            self.cache_bytes = sum(
                _cache_size_bytes(cache.past_key_values) + _tensor_size_bytes(cache.prefix_input_ids)
                for cache in self.row_prefix_caches
            )
            return

        console.print(
            "[cyan]SemanticHistogram[/cyan] cache prep: building sampled KV caches from "
            f"{len(self.row_texts):,} candidate rows."
        )
        self.selected_indices = self._select_representative_indices(
            image_embeddings=image_embeddings,
            n_components=n_components,
        )
        self.row_prefix_caches = []
        for row_index in self.selected_indices:
            prefix_text = self.backend.build_row_prefix(self.row_texts[row_index])
            self.row_prefix_caches.append(self.backend.build_prefix_cache(prefix_text))

        self.cache_bytes = sum(
            _cache_size_bytes(cache.past_key_values) + _tensor_size_bytes(cache.prefix_input_ids)
            for cache in self.row_prefix_caches
        )
        self._persist_cache_artifact(n_components=n_components)
        console.print(
            "[cyan]SemanticHistogram[/cyan] cache prep: saved KV state to "
            f"[bold]{artifact_path}[/bold]."
        )

    def estimate(self, predicate: str, predicate_embedding: Any) -> float:
        if self.image_embeddings is None:
            raise ValueError("Threshold estimator must be fitted before estimation.")
        if predicate_embedding is None:
            raise ValueError("Threshold estimation requires a predicate embedding.")
        if not self.row_prefix_caches:
            raise ValueError("No prepared row caches available for estimation.")

        positive_responses = 0
        runtime_suffix = self.backend.build_runtime_suffix(predicate)
        batch_size = max(1, min(self.eval_batch_size, len(self.row_prefix_caches)))
        for start in range(0, len(self.row_prefix_caches), batch_size):
            cache_batch = self.row_prefix_caches[start : start + batch_size]
            responses, stats = self.backend.evaluate_binary_batch(
                runtime_suffix,
                cache_batch,
            )
            self.cost_stats["llm_calls"] += stats["llm_calls"]
            self.cost_stats["model_batches"] += stats["model_batches"]
            self.cost_stats["tokens"] += stats["tokens"]
            positive_responses += sum(1 for result in responses if result == "1")

        selectivity = positive_responses / len(self.row_prefix_caches)
        similarities = (
            self.image_embeddings @ predicate_embedding.unsqueeze(1)
        ).view(-1)
        threshold = self.threshold_from_selectivity(
            similarities=similarities,
            selectivity=selectivity,
        )
        return float(threshold.item())

    def reset_cost_stats(self) -> None:
        self.cost_stats = {"usd": 0, "llm_calls": 0, "model_batches": 0, "tokens": 0}

    @staticmethod
    def build_cache_prefix(row_text: str) -> str:
        normalized_row_text = _normalize_whitespace(row_text)
        return (
            "Answer the following question based on the row text with '1' or '0'. "
            "Do not add any other comments.\n\n"
            f"Row text:\n{normalized_row_text}\n\n"
        )

    @staticmethod
    def threshold_from_selectivity(similarities: Any, selectivity: float) -> Any:
        torch = importlib.import_module("torch")
        clamped = min(max(selectivity, 0.0), 1.0)
        quantile = 1.0 - clamped
        return torch.quantile(similarities, quantile)

    def _load_cache_artifact(self, n_components: int) -> dict[str, Any] | None:
        artifact_path = self._cache_artifact_path(n_components=n_components)
        if not artifact_path.exists():
            return None

        torch = importlib.import_module("torch")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if artifact.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        return artifact

    def _persist_cache_artifact(self, n_components: int) -> None:
        artifact_path = self._cache_artifact_path(n_components=n_components)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

        torch = importlib.import_module("torch")
        torch.save(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "selected_indices": self.selected_indices,
                "row_prefix_caches": [
                    {
                        "prefix_length": cache.prefix_length,
                        "prefix_input_ids": cache.prefix_input_ids,
                        "past_key_values": cache.past_key_values,
                    }
                    for cache in self.row_prefix_caches
                ],
            },
            artifact_path,
        )

    def _cache_artifact_path(self, n_components: int) -> Path:
        dataset_tag = _sanitize_path_component(self.dataset_name)
        column_tag = _sanitize_path_component(self.source_column)
        model_tag = _sanitize_path_component(self.hf_model_name)
        fingerprint = self._dataset_fingerprint()
        filename = (
            f"{column_tag}__{fingerprint}__{model_tag}"
            f"__k{n_components}__s{self.seed}.pt"
        )
        return self.cache_dir / dataset_tag / filename

    def _dataset_fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.dataset_name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(self.source_column.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(self.hf_model_name.encode("utf-8"))
        hasher.update(b"\0")
        for row_text in self.row_texts:
            hasher.update(row_text.encode("utf-8", "replace"))
            hasher.update(b"\0")
        if self.image_embeddings is not None:
            hasher.update(self.image_embeddings.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]

    def _select_representative_indices(
        self,
        image_embeddings: Any,
        n_components: int,
    ) -> list[int]:
        sklearn_cluster = importlib.import_module("sklearn.cluster")
        KMeans = getattr(sklearn_cluster, "KMeans")
        torch = importlib.import_module("torch")

        total_rows = int(image_embeddings.shape[0])
        if total_rows == 0:
            return []

        target_count = max(1, min(total_rows, n_components))
        if target_count == total_rows:
            return list(range(total_rows))
        if target_count == 1:
            return [0]

        embeddings_np = image_embeddings.detach().cpu().numpy()
        kmeans = KMeans(
            n_clusters=target_count,
            random_state=self.seed,
            n_init="auto",
        )
        kmeans.fit(embeddings_np)
        centers = torch.from_numpy(kmeans.cluster_centers_).to(image_embeddings)
        distances = torch.cdist(centers, image_embeddings)

        selected: list[int] = []
        for row_distances in distances:
            for candidate_index in torch.argsort(row_distances).tolist():
                if candidate_index not in selected:
                    selected.append(int(candidate_index))
                    break

        return selected


class SemanticHistogram(AlgorithmInterface):
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

        self.embedding_model_key: str | None = None
        self.hf_model_name: str | None = None
        self.num_kv_caches: int | None = None
        self.kv_eval_batch_size: int | None = None
        self.seed = 42
        self.device = "cuda:0"
        self.max_new_tokens = 1
        self.cache_dir = Path("results/cache/semantic_histogram")

        self.prepared_states: dict[tuple[str, str], PreparedColumnState] = {}
        self.memory_consumption = 0
        self.cost_stats = {"usd": 0, "llm_calls": 0, "model_batches": 0, "tokens": 0}
        self._backend: LocalHFTextKVBackend | None = None
        self._threshold_based_cls = None
        self._full_estimator_cls = None

    def get_memory_consumption(self) -> int:
        return self.memory_consumption

    def get_cost_stats(self) -> dict:
        return self.cost_stats

    def reset_cost_stats(self) -> None:
        self.cost_stats = {"usd": 0, "llm_calls": 0, "model_batches": 0, "tokens": 0}
        for state in self.prepared_states.values():
            state.threshold_estimator.reset_cost_stats()

    def preparation(
        self,
        data_dfs: dict[str, pd.DataFrame],
        algorithm_kwargs: dict,
    ) -> None:
        console.print(
            "[cyan]SemanticHistogram[/cyan] preparation: resolving configuration and backend."
        )
        self.embedding_model_key = self._require_algorithm_kwarg(
            algorithm_kwargs,
            "embedding_model_key",
        )
        self.hf_model_name = self._require_algorithm_kwarg(
            algorithm_kwargs,
            "hf_model_name",
        )
        self.num_kv_caches = int(
            self._require_algorithm_kwarg(algorithm_kwargs, "num_kv_caches")
        )
        self.kv_eval_batch_size = int(
            algorithm_kwargs.get("kv_eval_batch_size", self.num_kv_caches)
        )
        self.device = str(algorithm_kwargs.get("device", self.device))
        self.seed = int(algorithm_kwargs.get("seed", self.seed))
        self.max_new_tokens = int(
            algorithm_kwargs.get("max_new_tokens", self.max_new_tokens)
        )
        self.cache_dir = Path(
            str(algorithm_kwargs.get("cache_dir", str(self.cache_dir)))
        )

        self._require_ml_dependencies()
        self._import_upstream_estimators()

        assert self.hf_model_name is not None
        console.print(
            "[cyan]SemanticHistogram[/cyan] preparation: initializing local HF model "
            f"[bold]{self.hf_model_name}[/bold] on [bold]{self.device}[/bold]."
        )
        self._backend = LocalHFTextKVBackend(
            model_name=self.hf_model_name,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
        )

        self.prepared_states = {}
        self.memory_consumption = 0

        console.print(
            "[cyan]SemanticHistogram[/cyan] preparation: scanning loaded datasets for "
            "supported text columns and embeddings."
        )
        for dataset_name, data_df in data_dfs.items():
            for source_column in SUPPORTED_SOURCE_COLUMNS:
                if source_column not in data_df.columns:
                    continue

                embedding_column = self.embedding_column_name(
                    source_column=source_column,
                    embedding_model_key=self.embedding_model_key,
                )
                if embedding_column not in data_df.columns:
                    continue

                row_texts, row_embeddings = self._extract_texts_and_embeddings(
                    data_df=data_df,
                    source_column=source_column,
                    embedding_column=embedding_column,
                )
                if not row_texts:
                    continue

                console.print(
                    "[cyan]SemanticHistogram[/cyan] preparation: building state for "
                    f"[bold]{dataset_name}[/bold].[bold]{source_column}[/bold] "
                    f"({len(row_texts):,}/{len(data_df):,} usable rows)."
                )
                state = self._build_state(
                    dataset_name=dataset_name,
                    source_column=source_column,
                    row_texts=row_texts,
                    row_embeddings=row_embeddings,
                    total_rows=int(data_df.shape[0]),
                )
                self.prepared_states[(dataset_name, source_column)] = state
                self.memory_consumption += state.memory_bytes

        console.print(
            "[cyan]SemanticHistogram[/cyan] preparation complete: "
            f"{len(self.prepared_states)} column state(s) ready."
        )

    def run(self, query_spec: QuerySpecification) -> CardinalityEstimate:
        if len(query_spec.datasets) != 1:
            return CardinalityEstimate.unsupported(
                "SemanticHistogram supports only single-table queries."
            )

        column_refs = [
            part.value
            for part in query_spec.filter_parsed.parts
            if part.type == QueryTemplatePartType.COLUMN_REF
        ]
        if len(column_refs) != 1:
            return CardinalityEstimate.unsupported(
                "SemanticHistogram supports only predicates with exactly one column reference."
            )

        dataset_name = query_spec.datasets[0].table_ref
        source_column = column_refs[0].column_name
        if self.embedding_model_key is None:
            raise RuntimeError("SemanticHistogram must be prepared before running.")

        state = self.prepared_states.get((dataset_name, source_column))
        if state is None:
            return CardinalityEstimate.unsupported(
                f"SemanticHistogram does not support dataset '{dataset_name}' and column '{source_column}'."
            )

        embedding_values = query_spec.embeddings.get(self.embedding_model_key)
        if embedding_values is None:
            raise ValueError(
                f"Query {query_spec.id} does not provide embedding key '{self.embedding_model_key}'."
            )

        torch = importlib.import_module("torch")
        query_embedding = torch.tensor(embedding_values, dtype=torch.float32)
        query_embedding = torch.nn.functional.normalize(query_embedding, dim=0, p=2)

        rewritten_question = self.rewrite_single_column_query(query_spec)
        raw_estimate = state.estimator.estimate(rewritten_question, query_embedding)
        rounded_estimate = int(max(0, round(float(raw_estimate))))

        self.cost_stats = dict(state.threshold_estimator.cost_stats)
        return CardinalityEstimate(CardinalityEstimateKind.INT, value=rounded_estimate)

    @staticmethod
    def _require_algorithm_kwarg(
        algorithm_kwargs: dict[str, Any],
        key: str,
    ) -> str:
        value = algorithm_kwargs.get(key)
        if value in (None, ""):
            raise ValueError(f"SemanticHistogram requires algorithm_kwargs['{key}'].")
        return str(value)

    @staticmethod
    def sanitize_embedding_model_key(embedding_model_key: str) -> str:
        normalized = embedding_model_key.lower().replace("/", "_")
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        return normalized.strip("_")

    @classmethod
    def embedding_column_name(
        cls,
        source_column: str,
        embedding_model_key: str,
    ) -> str:
        return f"{source_column}_embeddings_{cls.sanitize_embedding_model_key(embedding_model_key)}"

    @staticmethod
    def rewrite_single_column_query(query_spec: QuerySpecification) -> str:
        rewritten_parts = []
        for part in query_spec.filter_parsed.parts:
            if part.type == QueryTemplatePartType.TEXT:
                rewritten_parts.append(str(part.value))
            elif part.type == QueryTemplatePartType.COLUMN_REF:
                rewritten_parts.append("the row text shown above")

        rewritten = _normalize_whitespace("".join(rewritten_parts))
        if rewritten:
            return rewritten
        return "the row text shown above matches the query"

    @staticmethod
    def _ensure_upstream_path() -> Path:
        submodule_src_path = (
            Path(__file__).resolve().parent / "semantic_histograms" / "src"
        )
        if not submodule_src_path.exists():
            raise RuntimeError(
                "SemanticHistogram requires the semantic_histograms submodule at "
                f"'{submodule_src_path}'. Initialize submodules before running."
            )

        submodule_src = str(submodule_src_path)
        if submodule_src not in sys.path:
            sys.path.insert(0, submodule_src)
        return submodule_src_path

    @staticmethod
    def _require_ml_dependencies() -> None:
        missing_packages = []
        for package_name in ("torch", "transformers", "sklearn"):
            try:
                importlib.import_module(package_name)
            except ImportError:
                missing_packages.append(package_name)

        if missing_packages:
            joined = ", ".join(missing_packages)
            raise RuntimeError(
                "SemanticHistogram requires optional ML dependencies to be installed. "
                f"Missing packages: {joined}."
            )

    def _import_upstream_estimators(self) -> None:
        self._ensure_upstream_path()
        module = importlib.import_module("mmce.estimators.base_threshold_estimator")
        self._threshold_based_cls = getattr(module, "ThresholdBasedCardinalityEstimator")
        self._full_estimator_cls = getattr(module, "FullEstimator")

    @staticmethod
    def _extract_texts_and_embeddings(
        data_df: pd.DataFrame,
        source_column: str,
        embedding_column: str,
    ) -> tuple[list[str], Any]:
        valid_mask = data_df[source_column].notna() & data_df[embedding_column].notna()
        subset = data_df.loc[valid_mask, [source_column, embedding_column]]

        row_texts: list[str] = []
        row_embeddings: list[np.ndarray] = []

        for row_text, embedding in subset.itertuples(index=False, name=None):
            normalized_text = _normalize_whitespace(str(row_text))
            if not normalized_text:
                continue

            embedding_array = np.asarray(embedding, dtype=np.float32)
            if embedding_array.ndim != 1 or embedding_array.size == 0:
                continue
            if not np.isfinite(embedding_array).all():
                continue

            row_texts.append(normalized_text)
            row_embeddings.append(embedding_array)

        if not row_embeddings:
            torch = importlib.import_module("torch")
            return [], torch.empty((0, 0), dtype=torch.float32)

        torch = importlib.import_module("torch")
        embedding_matrix = torch.from_numpy(np.stack(row_embeddings)).to(torch.float32)
        embedding_matrix = torch.nn.functional.normalize(embedding_matrix, dim=1, p=2)
        return row_texts, embedding_matrix

    def _build_state(
        self,
        dataset_name: str,
        source_column: str,
        row_texts: list[str],
        row_embeddings: Any,
        total_rows: int,
    ) -> PreparedColumnState:
        if self._backend is None:
            raise RuntimeError("SemanticHistogram backend is not initialized.")
        if self._threshold_based_cls is None or self._full_estimator_cls is None:
            raise RuntimeError("SemanticHistogram upstream estimator classes are unavailable.")
        if self.num_kv_caches is None:
            raise RuntimeError("SemanticHistogram num_kv_caches is not configured.")
        if self.kv_eval_batch_size is None:
            raise RuntimeError("SemanticHistogram kv_eval_batch_size is not configured.")
        if self.hf_model_name is None:
            raise RuntimeError("SemanticHistogram HF model name is not configured.")

        effective_num_kv_caches = max(1, min(self.num_kv_caches, len(row_texts)))
        console.print(
            "[cyan]SemanticHistogram[/cyan] "
            f"{dataset_name}.{source_column}: preparing {effective_num_kv_caches} KV cache(s)."
        )
        threshold_estimator = TextPreloadedKVThresholdEstimator(
            dataset_name=dataset_name,
            source_column=source_column,
            row_texts=row_texts,
            backend=self._backend,
            cache_dir=self.cache_dir,
            hf_model_name=self.hf_model_name,
            num_kv_caches=effective_num_kv_caches,
            eval_batch_size=max(1, min(self.kv_eval_batch_size, effective_num_kv_caches)),
            seed=self.seed,
        )
        estimator = self._threshold_based_cls(
            threshold_estimator=threshold_estimator,
            threshold_to_cardinality=self._full_estimator_cls(),
            determines_bucket_sizes="threshold_estimator",
            other_bucket_size=int(row_embeddings.shape[0]),
        )
        estimator.fit(
            images=[],
            image_embeddings=row_embeddings,
            n_components=effective_num_kv_caches,
            seed=self.seed,
        )

        memory_bytes = self._estimate_state_memory(
            row_texts=row_texts,
            row_embeddings=row_embeddings,
            threshold_estimator=threshold_estimator,
        )
        return PreparedColumnState(
            dataset_name=dataset_name,
            source_column=source_column,
            estimator=estimator,
            threshold_estimator=threshold_estimator,
            total_rows=total_rows,
            valid_rows=len(row_texts),
            memory_bytes=memory_bytes,
        )

    @staticmethod
    def _estimate_state_memory(
        row_texts: list[str],
        row_embeddings: Any,
        threshold_estimator: TextPreloadedKVThresholdEstimator,
    ) -> int:
        text_bytes = sum(len(text.encode("utf-8")) for text in row_texts)
        embedding_bytes = int(row_embeddings.numel() * row_embeddings.element_size())
        return text_bytes + embedding_bytes + threshold_estimator.cache_bytes
