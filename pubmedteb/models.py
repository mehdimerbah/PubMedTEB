"""Model registry for PubMedTEB benchmark evaluation.

Maps short model names to HuggingFace model IDs and loading configuration.
All models are sentence-transformer compatible.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODELS: dict[str, dict] = {
    # ── Biomedical ──
    "pubmedbert": {
        "model_id": "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
    },
    "biolinkbert": {
        "model_id": "michiyasunaga/BioLinkBERT-large",
    },
    "medembed-small": {
        "model_id": "abhinand/MedEmbed-small-v0.1",
    },
    "medembed-large": {
        "model_id": "abhinand/MedEmbed-large-v0.1",
    },
    "biosimcse": {
        "model_id": "kamalkraj/BioSimCSE-BioLinkBERT-BASE",
    },
    # ── Scientific ──
    "specter2": {
        # SPECTER2 ships as an `adapters`-library adapter on top of specter2_base,
        # not a peft adapter — sentence_transformers' default loader trips on it.
        "loader": "specter2",
        "base_id": "allenai/specter2_base",
        "adapter_id": "allenai/specter2",
    },
    "scincl": {
        "model_id": "malteos/scincl",
    },
    # ── General baselines ──
    "e5-large": {
        "model_id": "intfloat/e5-large-v2",
        "encode_kwargs": {"normalize_embeddings": True},
    },
    "jina-v3": {
        # Uses custom remote code. Pinned to mteb's tested revision; later
        # snapshots break under transformers≥5 because XLMRobertaLoRA.__init__
        # skips post_init() and never sets `all_tied_weights_keys`.
        "model_id": "jinaai/jina-embeddings-v3",
        "model_kwargs": {
            "trust_remote_code": True,
            "revision": "215a6e121fa0183376388ac6b1ae230326bfeaed",
        },
    },
}


def _patch_jina_task_translation(model: SentenceTransformer) -> SentenceTransformer:
    """Translate sentence-transformers task names to jina-v3 LoRA task names.

    sentence-transformers ≥5 hardcodes `task='query'` / `task='document'` in
    encode_query/encode_document, but jina-v3's custom_st.forward rejects
    anything outside its task vocabulary. Wrap encode() to translate.
    """
    jina_tasks = {"query": "retrieval.query", "document": "retrieval.passage"}
    original_encode = model.encode

    def encode(*args, **kwargs):
        task = kwargs.get("task")
        if task in jina_tasks:
            kwargs["task"] = jina_tasks[task]
        return original_encode(*args, **kwargs)

    model.encode = encode
    return model


def _load_specter2(base_id: str, adapter_id: str) -> SentenceTransformer:
    """Load SPECTER2 (base + adapters-library adapter) as a SentenceTransformer.

    Builds a sentence-transformers pipeline manually: a Transformer module
    whose `auto_model` is replaced by an AutoAdapterModel with the SPECTER2
    adapter activated, followed by CLS pooling (per the model card).
    """
    from adapters import AutoAdapterModel
    from sentence_transformers.models import Pooling, Transformer

    transformer = Transformer(base_id, max_seq_length=512)
    adapter_model = AutoAdapterModel.from_pretrained(base_id)
    adapter_model.load_adapter(
        adapter_id, source="hf", load_as="specter2", set_active=True
    )
    transformer.auto_model = adapter_model

    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="cls")
    return SentenceTransformer(modules=[transformer, pooling])


def get_model(name_or_path: str) -> SentenceTransformer:
    """Load a model by registry name, HuggingFace ID, or local path.

    Args:
        name_or_path: Short name from MODELS registry, a HuggingFace model ID,
            or a local filesystem path to a model checkpoint.

    Returns:
        A loaded SentenceTransformer model.
    """
    if name_or_path in MODELS:
        entry = MODELS[name_or_path]
        if entry.get("loader") == "specter2":
            logger.info(
                "Loading model '%s' (%s + adapter %s)",
                name_or_path, entry["base_id"], entry["adapter_id"],
            )
            return _load_specter2(entry["base_id"], entry["adapter_id"])
        model_id = entry["model_id"]
        model_kwargs = entry.get("model_kwargs", {})
        logger.info("Loading model '%s' (%s)", name_or_path, model_id)
    else:
        model_id = name_or_path
        model_kwargs = {}
        logger.info("Loading model from '%s'", model_id)

    model = SentenceTransformer(model_id, **model_kwargs)
    if name_or_path == "jina-v3":
        model = _patch_jina_task_translation(model)
    return model


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(MODELS.keys())
