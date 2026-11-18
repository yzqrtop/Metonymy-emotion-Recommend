"""Data utilities: schema, text processing, seed dataset builder and loaders."""

from .schema import (  # noqa: F401
    EMOTION_TO_POLARITY,
    EMOTION_TYPES,
    ID_TO_POLARITY,
    ID_TO_TYPE,
    POLARITIES,
    POLARITY_TO_ID,
    TYPE_TO_ID,
    box_iou,
    dataset_statistics,
    group_split,
    read_json,
    read_jsonl,
    split_records,
    write_json,
    write_jsonl,
)
