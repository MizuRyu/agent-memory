# agent-memory

Claude Code に長期記憶を持たせるツールの試作です。セッションの終わりに会話を保存し、過去の文脈をハイブリッド検索で取り出します。

## 現状

2026-04-05 の commit のまま、更新していません。動作を確かめているのは、偽の埋め込みを使うシナリオのテスト 1 件だけです。

## 仕組み

1. 会話の記録（transcript の JSONL）を分割します。
2. [Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-310m) で埋め込みを作り、`~/.agent-memory/memory.db`（SQLite と sqlite-vec）に保存します。
3. 検索では、キーワード（SQLite FTS5）と埋め込みの両方で探して順位を合わせ（RRF）、今のプロジェクトの記憶と新しい記憶を重く扱います。

詳しくは [仕様](docs/SPEC.md) を参照してください。埋め込みモデルと検索手法の下調べは [docs/research/](docs/research/) にあります。

## 必要なもの

| 必要なもの | 用途 |
| --- | --- |
| Python 3.12 以上、[uv](https://docs.astral.sh/uv/) | 実行と依存の管理 |
| Ruri v3 のモデル（約 1.2GB） | 埋め込み。初回の保存か検索で自動でダウンロードされる |
| [jq](https://jqlang.org/) | hook で、Claude Code から渡される JSON を読む |

## 使い方

```bash
uv sync
```

| コマンド | すること |
| --- | --- |
| `agent-memory save --transcript <transcript.jsonl> --project <作業ディレクトリ>` | 会話を保存する。`--dry-run` を付けると、分割した結果を表示するだけで保存しない |
| `agent-memory search --query "認証設計の方針" --project <作業ディレクトリ> --top 5` | 過去の記憶を検索する |
| `agent-memory summary` | 保存した記憶の件数などを表示する |
| `agent-memory debug` | 記憶のデータベースを VS Code で開く |

### セッションの終わりに自動で保存する

`~/.claude/settings.json` に `SessionEnd` の hook を足します。Claude Code は transcript のパスを標準入力の JSON（`transcript_path`）で渡すので、jq で取り出して `save` に渡します。

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run --project /path/to/agent-memory agent-memory save --transcript \"$(jq -r .transcript_path)\" --project \"$CLAUDE_PROJECT_DIR\"",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

`SessionEnd` の hook は既定で 1.5 秒しか待たれないため、`timeout` で延ばしています（上限は 60 秒）。モデルの読み込みと埋め込みが 60 秒を超えると、保存されずに終わります。

## テスト

```bash
uv sync --extra dev
uv run python -m pytest tests/ -v
```

テストは偽の埋め込みを使うので、モデルはダウンロードしません。

## 参考

次の Zenn の記事を参考にしています。

- [noprogllama さんの Zenn の記事](https://zenn.dev/noprogllama/articles/7c24b2c2410213)
