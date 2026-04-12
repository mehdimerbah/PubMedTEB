"""Dataset builder registry."""

from __future__ import annotations

from pathlib import Path

from pubmedteb.builders.base import DatasetBuilder
from pubmedteb.builders.title_retrieval import TitleRetrievalBuilder

BUILDERS: dict[str, type[DatasetBuilder]] = {
    "title-retrieval": TitleRetrievalBuilder,
}


def get_builder(
    task_name: str,
    output_dir: Path | None = None,
    **kwargs,
) -> DatasetBuilder:
    """Instantiate a dataset builder by task name.

    Args:
        task_name: Task slug (e.g., "title-retrieval").
        output_dir: Override output directory. Defaults to ``datasets/pubmed_{slug}``.
        **kwargs: Passed to the builder constructor (seed, size, etc.).
    """
    if task_name not in BUILDERS:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(BUILDERS)}"
        )
    if output_dir is None:
        slug = task_name.replace("-", "_")
        output_dir = Path(f"datasets/pubmed_{slug}")
    return BUILDERS[task_name](output_dir=output_dir, **kwargs)
