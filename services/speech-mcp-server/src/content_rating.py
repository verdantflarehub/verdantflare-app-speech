"""Re-export speech_mcp.content_rating for direct src.content_rating imports."""
from __future__ import annotations

from speech_mcp.content_rating import (
    CONTENT_RATING_SCHEMA_SQL,
    ContentRatingError,
    ContentRatingStore,
    _ClosingConnection,
    default_unrated_dict,
    include_nsfw_from_headers,
    normalize_reasons,
)

__all__ = [
    "CONTENT_RATING_SCHEMA_SQL",
    "ContentRatingError",
    "ContentRatingStore",
    "_ClosingConnection",
    "default_unrated_dict",
    "include_nsfw_from_headers",
    "normalize_reasons",
]
