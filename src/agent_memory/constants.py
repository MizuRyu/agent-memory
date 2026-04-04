from pathlib import Path

# Database
DEFAULT_DB_PATH = Path.home() / ".agent-memory" / "memory.db"

# Embedding
DIMENSION = 768
DOCUMENT_PREFIX = "検索文書: "
QUERY_PREFIX = "検索クエリ: "
DEFAULT_MODEL_NAME = "cl-nagoya/ruri-v3-310m"

# Chunker
MAX_CHUNK_CHARS = 32_000  # ~8192 tokens (conservative estimate for Japanese)

# Search
RRF_K = 60
PROJECT_WEIGHT_SAME = 1.0
PROJECT_WEIGHT_DIFF = 0.3
DECAY_HALF_LIFE_DAYS = 30
