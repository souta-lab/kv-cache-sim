"""kv-cache-sim のコマンドラインインターフェース。

使い方:
    python -m kv_cache_sim calc --layers 32 --hidden 4096 --heads 32 --kv-heads 8 --seq 8192
    python -m kv_cache_sim compare examples/compare.json --seq 8192
"""

from __future__ import annotations

import argparse
import json
import sys

from .core import (
    ModelConfig,
    effective_seq_len,
    format_bytes,
    max_seq_len_for_budget,
    per_token_bytes,
    per_token_bytes_per_layer,
    total_bytes,
)


def _config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        layers=args.layers,
        hidden_size=args.hidden,
        num_heads=args.heads,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        attention=args.attention,
        kv_lora_rank=args.kv_lora_rank,
        rope_head_dim=args.rope_head_dim,
        sliding_window=args.sliding_window,
        name=getattr(args, "name", "custom"),
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--layers", type=int, required=True, help="層数")
    parser.add_argument("--hidden", type=int, required=True, help="隠れ次元")
    parser.add_argument("--heads", type=int, required=True, help="Q ヘッド数")
    parser.add_argument("--kv-heads", type=int, default=None, help="KV ヘッド数 (省略時は Q ヘッド数と同じ)")
    parser.add_argument("--head-dim", type=int, default=None, help="ヘッド次元 (省略時は hidden / heads)")
    parser.add_argument(
        "--attention",
        choices=("mha", "gqa", "mqa", "mla"),
        default="gqa",
        help="注意機構の種別",
    )
    parser.add_argument("--kv-lora-rank", type=int, default=None, help="MLA の圧縮ランク")
    parser.add_argument("--rope-head-dim", type=int, default=64, help="MLA の RoPE 用キー次元")
    parser.add_argument("--sliding-window", type=int, default=None, help="スライディングウィンドウ幅")
    parser.add_argument("--bits", type=int, default=16, help="KV の量子化ビット数 (既定: 16 = fp16/bf16)")
    parser.add_argument("--batch", type=int, default=1, help="バッチサイズ")


def cmd_calc(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    per_layer = per_token_bytes_per_layer(cfg, args.bits)
    per_token = per_token_bytes(cfg, args.bits)
    eff_seq = effective_seq_len(cfg, args.seq)
    total = total_bytes(cfg, args.seq, args.batch, args.bits)

    print(f"構成: {args.attention} / {cfg.layers} 層")
    if args.attention == "mla":
        print(f"  圧縮ランク: {cfg.kv_lora_rank}, RoPE 次元: {cfg.rope_head_dim}")
    else:
        print(
            f"  Q ヘッド: {cfg.num_heads}, KV ヘッド: {cfg.resolved_kv_heads()}, "
            f"ヘッド次元: {cfg.resolved_head_dim()}"
        )
    print(f"KV 精度: {args.bits} bit/要素")
    print(f"1 トークン・1 層: {format_bytes(per_layer)}")
    print(f"1 トークン (全層): {format_bytes(per_token)}")
    if eff_seq != args.seq:
        print(f"実効系列長: {eff_seq} (要求 {args.seq}, 窓幅 {cfg.sliding_window})")
    print(
        f"合計 (系列長 {args.seq}, バッチ {args.batch}): {format_bytes(total)}"
    )
    if args.budget_gib is not None:
        budget = args.budget_gib * (1024**3)
        n = max_seq_len_for_budget(cfg, budget, args.batch, args.bits)
        print(f"予算 {args.budget_gib} GiB に収まる最大系列長: {n} トークン")
    return 0


def _config_from_dict(d: dict) -> ModelConfig:
    return ModelConfig(
        layers=d["layers"],
        hidden_size=d["hidden_size"],
        num_heads=d["num_heads"],
        num_kv_heads=d.get("num_kv_heads"),
        head_dim=d.get("head_dim"),
        attention=d.get("attention", "gqa"),
        kv_lora_rank=d.get("kv_lora_rank"),
        rope_head_dim=d.get("rope_head_dim", 64),
        sliding_window=d.get("sliding_window"),
        name=d.get("name", "custom"),
    )


def cmd_compare(args: argparse.Namespace) -> int:
    with open(args.file, encoding="utf-8") as f:
        data = json.load(f)
    entries = data["configs"] if isinstance(data, dict) else data
    configs = [_config_from_dict(d) for d in entries]

    header = f"{'name':<28}{'per token':>14}{'total':>16}"
    print(f"系列長 {args.seq}, バッチ {args.batch}, KV 精度 {args.bits} bit")
    print(header)
    print("-" * len(header))
    for cfg in configs:
        per = per_token_bytes(cfg, args.bits)
        tot = total_bytes(cfg, args.seq, args.batch, args.bits)
        print(f"{cfg.name:<28}{format_bytes(per):>14}{format_bytes(tot):>16}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kv-cache-sim",
        description="Transformer の KV キャッシュ使用量シミュレータ",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_calc = sub.add_parser("calc", help="1 構成分の KV キャッシュ量を計算する")
    _add_config_args(p_calc)
    p_calc.add_argument("--seq", type=int, required=True, help="系列長 (トークン数)")
    p_calc.add_argument(
        "--budget-gib", type=float, default=None, help="メモリ予算 (GiB)。指定すると収まる最大系列長も出す"
    )
    p_calc.set_defaults(func=cmd_calc)

    p_cmp = sub.add_parser("compare", help="JSON の複数構成を同じ条件で比較する")
    p_cmp.add_argument("file", help="構成を並べた JSON ファイル")
    p_cmp.add_argument("--seq", type=int, default=8192, help="系列長 (既定: 8192)")
    p_cmp.add_argument("--batch", type=int, default=1, help="バッチサイズ")
    p_cmp.add_argument("--bits", type=int, default=16, help="KV の量子化ビット数")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
