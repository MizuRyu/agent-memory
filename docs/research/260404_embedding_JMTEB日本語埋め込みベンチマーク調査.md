**調査日:** 2026-04-04

# JMTEB（Japanese Massive Text Embedding Benchmark）調査

## 概要

JMTEB は、日本語テキスト埋め込みモデルを統一的に評価するためのベンチマーク。英語圏の [MTEB](https://github.com/embeddings-benchmark/mteb) に相当する日本語版として、[SB Intuitions](https://www.sbintuitions.co.jp/) が開発・公開している。

日本語埋め込みモデルは多数存在するが、これまで標準的な評価基盤がなく、モデル間の横断比較が困難だった。JMTEB はこの課題を解決し、RAG や検索システムにおけるモデル選定の判断材料を提供する。

- GitHub: [sbintuitions/JMTEB](https://github.com/sbintuitions/JMTEB)
- HuggingFace: [sbintuitions/JMTEB](https://huggingface.co/datasets/sbintuitions/JMTEB)
- ライセンス: CC-BY-SA-4.0

## バージョン

### v1.x（初期版）

6タスク・16データセットで構成。独自の評価スクリプトで動作。

### v2.0（現行版）

MTEBフレームワークと統合され、**5タスク・28データセット**に拡張。MTEBのリーダーボード・エコシステムとの互換性を実現。

## タスク構成（v2.0: 28データセット）

| タスク | データセット数 | 内容 | 評価指標 |
|--------|---------------|------|----------|
| Classification | 7 | テキスト分類（感情分析、意図分類等） | Accuracy |
| Clustering | 3 | 意味的に類似したテキストのグループ化 | V-measure |
| STS | 2 | 文ペアの意味的類似度 | Spearman相関 |
| Retrieval | 11 | クエリに対する関連文書の検索 | nDCG@10 |
| Reranking | 5 | 検索結果の関連度順位付け | MAP |

### 主要データセット

- **Classification**: Amazon Reviews、MASSIVE Intent/Scenario、日本語感情分析、SIB200、WRIME
- **Clustering**: Livedoor News、MewsC-16-ja、SIB200
- **STS**: JSTS、JSICK
- **Retrieval**: JAQKET、Mr.TyDi、JaGovFAQs、MIRACL、JaCWIR、Mintaka、MultiLongDoc
- **Reranking**: ESCI、JQaRA、JaCWIR、MIRACL、MultiLongDoc

## 主要モデルの性能比較

### 総合ランキング（JMTEB平均スコア）

| モデル | パラメータ | 次元 | 平均 | 備考 |
|--------|-----------|------|------|------|
| **Ruri v3 (310M)** | 315M | 768 | **77.24** | 日本語SOTA、ModernBERT-Jaベース |
| PLaMo-Embedding-1B | 1.05B | - | 76.10 | PFN開発、1Bクラス |
| Ruri v3 (130M) | 132M | 512 | 76.55 | 軽量で高性能 |
| Sarashina-embedding-v1-1b | 1.22B | - | 75.50 | SB Intuitions開発 |
| Ruri v3 (70M) | 70M | 384 | 75.48 | |
| Ruri Large v2 | 337M | 1024 | 74.55 | 旧世代 |
| Ruri v3 (30M) | 37M | 256 | 74.51 | 最小でもtext-embedding-3-large超え |
| OpenAI text-embedding-3-large | 非公開 | 3072 | 73.97 | API型、高コスト |
| Amber-large | 315M | - | 73.22 | レトリバ開発 |
| pkshatech/GLuCoSE-base-ja | 133M | 768 | - | 旧世代の日本語モデル上位 |

### タスク別スコア（Ruri v3-310M）

| タスク | スコア |
|--------|--------|
| Reranking | 93.43 |
| Retrieval | 81.89 |
| STS | 81.22 |
| Classification | 78.66 |
| PairClassification | 62.60 |
| Clustering | 55.69 |
| **平均** | **77.24** |

### Ruri v3 シリーズ詳細

[cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m) — 名古屋大学が開発した日本語特化埋め込みモデル。

- **ベースモデル**: ModernBERT-Ja（FlashAttention対応で高速推論）
- **最大系列長**: 8,192トークン（v2の512から大幅拡張）
- **語彙サイズ**: 100Kトークン（分かち書き器不要）
- **サイズ展開**: 30M / 70M / 130M / 310M の4バリエーション
- **特筆点**: 315Mパラメータで1B超モデル（PLaMo-1B, Sarashina-1B）を上回る

### 注目ポイント

- **Ruri v3 が日本語 SOTA**: 平均77.24で既存SOTAの75.5を大きく更新。特に Retrieval（81.89）が強い
- **小型モデルの健闘**: Ruri v3-30M（37Mパラメータ）でも OpenAI text-embedding-3-large を上回る
- **Qwen3-Embedding は日本語が弱い**: 0.6Bパラメータだが日本語タスクは振るわない結果（[検証記事](https://secon.dev/entry/2025/06/11/100000-qwen3-embedding-jmteb/)）

※ 最新のリーダーボードは [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) で確認可能（Japaneseフィルタ使用）。

## JMTEB-lite（軽量版）

JMTEBのフル評価はコーパスが大きく時間がかかるため、検索・リランキングタスクのコーパスを大幅に削減した軽量版が用意されている。

- **速度**: JMTEBの約5倍高速
- **信頼性**: フル版と高い相関を維持
- **互換性**: JMTEBの評価スクリプトとそのまま互換
- **推奨利用**: 開発中のアジャイル評価にはJMTEB-lite、最終評価にはJMTEBフル版
- HuggingFace: [sbintuitions/JMTEB-lite](https://huggingface.co/datasets/sbintuitions/JMTEB-lite)

## 使い方

### インストール

```bash
git clone https://github.com/sbintuitions/JMTEB.git
cd JMTEB
poetry install
```

### Python API（v2.0）

```python
from jmteb.v2 import JMTEBModel, JMTEBV2Evaluator
from jmteb.v2.tasks import get_jmteb_tasks

model = JMTEBModel.from_sentence_transformer("cl-nagoya/ruri-v3-30m")
tasks = get_jmteb_tasks()
evaluator = JMTEBV2Evaluator(model=model, tasks=tasks)
results = evaluator.run()
```

### CLI

```bash
python -m jmteb.v2 --model_name cl-nagoya/ruri-v3-30m --save_path results_v2
```

### データセット読み込み

```python
from datasets import load_dataset
dataset = load_dataset("sbintuitions/JMTEB", name="livedoor_news", split="train")
```

## 正規化の影響

SB Intuitionsの分析によると、ベクトルの正規化（L2ノルム=1に変換）は:

- **Classification・Retrieval**: スコア低下
- **STS・Clustering**: 影響なし

埋め込みのスケール情報がタスクによっては重要であることを示唆している。

## 論文

「JMTEB and JMTEB-lite: Japanese Massive Text Embedding Benchmark and Its Lightweight Version」

- 著者: Li Shengzhe ほか（SB Intuitions）
- 発表: LREC 2026（2026年5月、スペイン・パルマ）
- NL研 優秀研究賞 受賞

**最終更新日:** 2026-04-04