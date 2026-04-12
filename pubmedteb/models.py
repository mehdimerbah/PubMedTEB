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
        "model_id": "Henrychur/MedEmbed-small-v0.1",
    },
    "medembed-large": {
        "model_id": "Henrychur/MedEmbed-large-v0.1",
    },
    "biosimcse": {
        "model_id": "kamalkraj/BioSimCSE-BioLinkBERT-BASE",
    },
    # ── Scientific ──
    "specter2": {
        "model_id": "allenai/specter2",
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
        "model_id": "jinaai/jina-embeddings-v3",
    },
}


def get_model(name_or_path: str) -> SentenceTransformer:
    """Load a model by registry name, HuggingFace ID, or local path.

    Args:
        name_or_path: Short name from MODELS registry, a HuggingFace model ID,
            or a local filesystem path to a model checkpoint.

    Returns:
        A loaded SentenceTransformer model.
    """
    if name_or_path in MODELS:
        model_id = MODELS[name_or_path]["model_id"]
        logger.info("Loading model '%s' (%s)", name_or_path, model_id)
    else:
        model_id = name_or_path
        logger.info("Loading model from '%s'", model_id)

    return SentenceTransformer(model_id)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(MODELS.keys())
