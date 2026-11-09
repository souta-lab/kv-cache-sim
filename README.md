# kv-cache-sim

Transformer の KV キャッシュがどれだけメモリを食うかを、モデルの構成パラメータから計算するシミュレータです。外部依存なしの Python CLI です。

対応する注意機構:

- **MHA** (Multi-Head Attention)
- **GQA** (Grouped-Query Attention)
- **MQA** (Multi-Query Attention)
- **MLA** (Multi-head Latent Attention)
- スライディングウィンドウ注意（実効系列長を窓幅で頭打ち）
- KV の量子化（`--bits 8` / `--bits 4` など任意のビット数）
- メモリ予算から収まる最大系列長の逆算（`--budget-gib`）

## 計算式

MHA / GQA / MQA（1 トークン・1 層あたり）:

```
2 (K と V) × KVヘッド数 × ヘッド次元 × (ビット数 / 8) バイト
```

MLA（1 トークン・1 層あたり）:

```
(圧縮ランク kv_lora_rank + RoPE次元) × (ビット数 / 8) バイト
```

MLA では圧縮済みの潜在ベクトルと RoPE 用キーを全ヘッドで共有して保持するため、キャッシュ量はヘッド数に依存しません。

総量は `1トークンあたり × 層数 × 実効系列長 × バッチサイズ` です。スライディングウィンドウがある構成では実効系列長が `min(系列長, 窓幅)` になります。

## 使い方

インストールせずに、リポジトリ直下からそのまま実行できます。

```console
$ PYTHONPATH=src python3 -m kv_cache_sim calc \
    --layers 32 --hidden 4096 --heads 32 --kv-heads 8 \
    --seq 8192 --budget-gib 24
構成: gqa / 32 層
  Q ヘッド: 32, KV ヘッド: 8, ヘッド次元: 128
KV 精度: 16 bit/要素
1 トークン・1 層: 4.00 KiB
1 トークン (全層): 128.00 KiB
合計 (系列長 8192, バッチ 1): 1.00 GiB
予算 24.0 GiB に収まる最大系列長: 196608 トークン
```

MLA の例:

```console
$ PYTHONPATH=src python3 -m kv_cache_sim calc \
    --layers 60 --hidden 7168 --heads 128 --attention mla \
    --kv-lora-rank 512 --seq 8192
構成: mla / 60 層
  圧縮ランク: 512, RoPE 次元: 64
KV 精度: 16 bit/要素
1 トークン・1 層: 1.12 KiB
1 トークン (全層): 67.50 KiB
合計 (系列長 8192, バッチ 1): 540.00 MiB
```

複数構成の比較は JSON に構成を並べて `compare` に渡します（サンプル: `examples/compare.json`、パラメータはすべて例示値）。

```console
$ PYTHONPATH=src python3 -m kv_cache_sim compare examples/compare.json --seq 131072
系列長 131072, バッチ 1, KV 精度 16 bit
name                             per token           total
----------------------------------------------------------
MHA 例 (32層, 32 KV heads)        512.00 KiB       64.00 GiB
GQA 例 (32層, 8 KV heads)         128.00 KiB       16.00 GiB
MQA 例 (32層, 1 KV head)           16.00 KiB        2.00 GiB
GQA + sliding window 4096       128.00 KiB       512.00 MiB
MLA 例 (60層, rank 512)            67.50 KiB        8.44 GiB
```

`pip install .` すると `kv-cache-sim` コマンドとしても使えます。

## 開発

テストは標準ライブラリの unittest で書かれています。依存のインストールは不要です。

```console
$ PYTHONPATH=src python3 -m unittest discover -s tests
```

## 構成

- `src/kv_cache_sim/core.py` — 計算だけを行うモジュール
- `src/kv_cache_sim/cli.py` — `calc` / `compare` サブコマンド
- `examples/compare.json` — 比較用のサンプル構成
- `tests/test_core.py` — 既知の参照値との照合を含むテスト

## 今後の予定（未実装）

- 実在モデルの構成プリセット集
- 系列長を横軸にした使用量グラフの出力
- ページング（ブロック単位の割り当て）を考慮した実効使用量の見積もり

## ライセンス

MIT License. `LICENSE` を参照してください。
