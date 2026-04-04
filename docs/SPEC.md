# agent-memory 仕様

## 設計原則

1. 外部サービスに依存しない（SQLite 1ファイルに完結）
2. バックグラウンドでトークンを消費しない（LLM不使用で保存）
3. セッション終了時に自動保存（Hook連携、手動操作ゼロ）
4. 1DB方式 + 検索時フィルタリング（保存時に振り分けない）

---

## コンポーネント構成

```
src/agent_memory/
├── chunker.py       # 会話 → Q&Aチャンク分割
├── embedder.py      # Ruri v3 によるベクトル化
├── store.py         # SQLite 保存（FTS5 + sqlite-vec）
├── search.py        # ハイブリッド検索 + RRF + 時間減衰 + PJ重み
└── cli.py           # CLI エントリポイント（save / search）

データ配置:
  ~/.agent-memory/memory.db  # プロジェクト横断の単一DB
```

---

## SQLite スキーマ

### メインテーブル

```sql
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    project_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime'))
);
```

### FTS5（キーワード検索用）

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    chunk_text,
    content='memories',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, chunk_text) VALUES ('delete', old.id, old.chunk_text);
END;
```

### sqlite-vec（ベクトル検索用）

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    memory_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);
```

### インデックス

```sql
CREATE INDEX IF NOT EXISTS idx_memories_project_path ON memories(project_path);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
```

---

## チャンカー

- **入力**: Claude Code の transcript ファイル（JSONL）
- **出力**: Q&A形式チャンクのリスト
- **方式**: ルールベース。ユーザー発言 + アシスタント応答 = 1チャンク
- **フィルタ**: `isMeta`, `tool_use`, `tool_result`, `thinking`, `file-history-snapshot` はスキップ
- **最大サイズ**: Ruri v3 の最大系列長 8192 トークンを目安。超過時は段落で分割

---

## エンベッダー

- **モデル**: `cl-nagoya/ruri-v3-310m`（768次元、Apache 2.0）
- **ライブラリ**: sentence-transformers
- **プレフィックス**: 保存時 `"検索文書: "`, 検索時 `"検索クエリ: "`
- **実行環境**: CPU（GPU不要）
- **インタフェース**: `Embedder` Protocol でテスト時に差し替え可能

---

## 検索エンジン

### スコア計算

```
最終スコア = RRF_score × project_weight × decay
```

| 要素 | 計算式 |
|------|--------|
| RRF | `Σ 1/(60 + rank_i)` |
| プロジェクト重み | 同一PJ: 1.0 / 別PJ: 0.3 |
| 時間減衰 | `0.5^(経過日数 / 30)` |

### 検索フロー

1. FTS5 キーワード検索（trigram）→ 上位N件 + 順位
2. sqlite-vec ベクトル類似検索 → 上位N件 + 順位
3. RRF でスコア統合
4. project_weight を適用
5. 時間減衰を適用
6. 上位N件を返却

---

## CLI

typer を使用。

### save

```bash
agent-memory save --transcript <path> --project <path>
```

### search

```bash
agent-memory search --query <text> [--project <path>] [--top <n>]
```

---

## Claude Code Hook 連携

### 保存（セッション終了時）

```json
{
  "hooks": {
    "PostSessionEnd": [
      {
        "command": "uv run --project /path/to/agent-memory save --transcript \"$TRANSCRIPT_PATH\" --project \"$PWD\""
      }
    ]
  }
}
```

### Agent への指示（CLAUDE.md）

```markdown
## 長期記憶
ユーザーの発言に関連する過去の文脈を検索してください:
`agent-memory search --query "<ユーザーの発言>" --top 5`
検索結果がある場合は、その文脈を踏まえて応答してください。
```

---

## 技術スタック

```
Python 3.12
uv
typer
SQLite + FTS5 (trigram) + sqlite-vec
sentence-transformers >=3.0
cl-nagoya/ruri-v3-310m (768次元, Apache 2.0)
```
