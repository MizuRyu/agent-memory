# agent-memory

Claude Code に長期記憶を持たせるツール。セッション終了時に会話を自動保存し、過去の文脈をハイブリッド検索で取り出す。

## セットアップ

```bash
uv sync
```

Ruri v3 モデル（約1.2GB）は初回実行時に自動ダウンロードされる。

## 使い方

### 保存

```bash
agent-memory save --transcript <transcript.jsonl> --project <作業ディレクトリ>
```

### 検索

```bash
agent-memory search --query "認証設計の方針" --project /path/to/project --top 5
```

## Claude Code Hook 設定

`~/.claude/settings.json` に追加:

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

## テスト

```bash
uv run python -m pytest tests/ -v
```

## 仕様

[docs/SPEC.md](docs/SPEC.md)


## 参考
以下のzenn記事を参考にさせていただいています。
- [noprogllamaさんのzenn記事](https://zenn.dev/noprogllama/articles/7c24b2c2410213)