"""KV キャッシュのメモリ使用量を計算する純粋計算モジュール。

対応する注意機構:
- MHA (Multi-Head Attention): KV ヘッド数 = Q ヘッド数
- GQA (Grouped-Query Attention): KV ヘッド数を Q ヘッド数で割ったグループ数分だけ保持
- MQA (Multi-Query Attention): KV ヘッド数 = 1
- MLA (Multi-head Latent Attention): KV を低ランクの潜在ベクトルに圧縮して保持

計算式 (MHA/GQA/MQA, 1トークン・1層あたり):
    2 (K と V) * num_kv_heads * head_dim * (bits / 8) バイト

計算式 (MLA, 1トークン・1層あたり):
    (kv_lora_rank + rope_head_dim) * (bits / 8) バイト
    ※ 圧縮済み潜在ベクトルと、切り離された RoPE 用キーを全ヘッド共有で保持するため、
      ヘッド数には依存しない。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """KV キャッシュ計算に必要なモデル構成。"""

    layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int | None = None  # None のとき num_heads と同じ (MHA)
    head_dim: int | None = None  # None のとき hidden_size // num_heads
    attention: str = "gqa"  # "mha" | "gqa" | "mqa" | "mla"
    kv_lora_rank: int | None = None  # MLA のとき必須
    rope_head_dim: int = 64  # MLA の RoPE 用キー次元
    sliding_window: int | None = None  # スライディングウィンドウ注意の窓幅
    name: str = "custom"

    def resolved_kv_heads(self) -> int:
        if self.attention == "mqa":
            return 1
        if self.num_kv_heads is not None:
            return self.num_kv_heads
        return self.num_heads

    def resolved_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size が num_heads で割り切れません。"
                "head_dim を明示してください。"
            )
        return self.hidden_size // self.num_heads

    def validate(self) -> None:
        if self.layers <= 0:
            raise ValueError("layers は正の整数にしてください")
        if self.num_heads <= 0:
            raise ValueError("num_heads は正の整数にしてください")
        if self.attention not in ("mha", "gqa", "mqa", "mla"):
            raise ValueError(f"未知の attention 種別です: {self.attention}")
        if self.attention == "mla":
            if self.kv_lora_rank is None or self.kv_lora_rank <= 0:
                raise ValueError("MLA では kv_lora_rank が必須です")
        else:
            kv = self.resolved_kv_heads()
            if kv <= 0 or kv > self.num_heads:
                raise ValueError("num_kv_heads は 1 以上 num_heads 以下にしてください")
            if self.num_heads % kv != 0:
                raise ValueError("num_heads は num_kv_heads で割り切れる必要があります")
            self.resolved_head_dim()  # 割り切れなければここで例外
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window は正の整数にしてください")


def bytes_per_element(bits: int) -> float:
    """量子化ビット数から 1 要素あたりのバイト数を返す。"""
    if bits <= 0:
        raise ValueError("bits は正の整数にしてください")
    return bits / 8.0


def per_token_bytes_per_layer(cfg: ModelConfig, bits: int = 16) -> float:
    """1 トークン・1 層あたりの KV キャッシュサイズ (バイト)。"""
    cfg.validate()
    bpe = bytes_per_element(bits)
    if cfg.attention == "mla":
        return (cfg.kv_lora_rank + cfg.rope_head_dim) * bpe
    return 2 * cfg.resolved_kv_heads() * cfg.resolved_head_dim() * bpe


def per_token_bytes(cfg: ModelConfig, bits: int = 16) -> float:
    """1 トークンあたりの KV キャッシュサイズ (バイト、全層分)。"""
    return per_token_bytes_per_layer(cfg, bits) * cfg.layers


def effective_seq_len(cfg: ModelConfig, seq_len: int) -> int:
    """スライディングウィンドウを考慮した実効系列長。"""
    if seq_len <= 0:
        raise ValueError("seq_len は正の整数にしてください")
    if cfg.sliding_window is not None:
        return min(seq_len, cfg.sliding_window)
    return seq_len


def total_bytes(
    cfg: ModelConfig, seq_len: int, batch: int = 1, bits: int = 16
) -> float:
    """系列長とバッチサイズに対する KV キャッシュ総量 (バイト)。"""
    if batch <= 0:
        raise ValueError("batch は正の整数にしてください")
    return per_token_bytes(cfg, bits) * effective_seq_len(cfg, seq_len) * batch


def max_seq_len_for_budget(
    cfg: ModelConfig, budget_bytes: float, batch: int = 1, bits: int = 16
) -> int:
    """メモリ予算内に収まる最大系列長 (トークン数)。

    スライディングウィンドウがある構成では窓幅が上限になる。
    """
    if budget_bytes <= 0:
        raise ValueError("budget_bytes は正の値にしてください")
    if batch <= 0:
        raise ValueError("batch は正の整数にしてください")
    per = per_token_bytes(cfg, bits) * batch
    n = int(budget_bytes // per)
    if cfg.sliding_window is not None:
        n = min(n, cfg.sliding_window)
    return n


def format_bytes(n: float) -> str:
    """バイト数を人間可読な文字列にする (1024 進数)。"""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(value) < 1024.0 or unit == "PiB":
            if unit == "B":
                return f"{value:.0f} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")
