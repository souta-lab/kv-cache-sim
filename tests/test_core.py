import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kv_cache_sim.core import (  # noqa: E402
    ModelConfig,
    effective_seq_len,
    format_bytes,
    max_seq_len_for_budget,
    per_token_bytes,
    per_token_bytes_per_layer,
    total_bytes,
)


class TestPerToken(unittest.TestCase):
    def test_mha_single_layer(self):
        # 2 (K,V) * 4 kv heads * 64 head_dim * 2 bytes = 1024
        cfg = ModelConfig(layers=1, hidden_size=256, num_heads=4, attention="mha")
        self.assertEqual(per_token_bytes_per_layer(cfg, bits=16), 1024)

    def test_gqa_reduces_cache(self):
        mha = ModelConfig(layers=32, hidden_size=4096, num_heads=32, attention="mha")
        gqa = ModelConfig(
            layers=32, hidden_size=4096, num_heads=32, num_kv_heads=8, attention="gqa"
        )
        self.assertEqual(per_token_bytes(gqa), per_token_bytes(mha) / 4)

    def test_mqa_equals_single_kv_head(self):
        mqa = ModelConfig(layers=8, hidden_size=512, num_heads=8, attention="mqa")
        gqa1 = ModelConfig(
            layers=8, hidden_size=512, num_heads=8, num_kv_heads=1, attention="gqa"
        )
        self.assertEqual(per_token_bytes(mqa), per_token_bytes(gqa1))

    def test_bits_halve_bytes(self):
        cfg = ModelConfig(
            layers=32, hidden_size=4096, num_heads=32, num_kv_heads=8, attention="gqa"
        )
        self.assertEqual(per_token_bytes(cfg, bits=8), per_token_bytes(cfg, bits=16) / 2)
        self.assertEqual(per_token_bytes(cfg, bits=4), per_token_bytes(cfg, bits=16) / 4)

    def test_mla_is_head_independent(self):
        # (512 + 64) * 2 bytes = 1152 per token per layer, regardless of heads
        cfg = ModelConfig(
            layers=1,
            hidden_size=7168,
            num_heads=128,
            attention="mla",
            kv_lora_rank=512,
            rope_head_dim=64,
        )
        self.assertEqual(per_token_bytes_per_layer(cfg, bits=16), 1152)

    def test_mla_requires_rank(self):
        cfg = ModelConfig(layers=1, hidden_size=1024, num_heads=16, attention="mla")
        with self.assertRaises(ValueError):
            per_token_bytes(cfg)

    def test_invalid_kv_heads(self):
        cfg = ModelConfig(
            layers=1, hidden_size=512, num_heads=8, num_kv_heads=16, attention="gqa"
        )
        with self.assertRaises(ValueError):
            per_token_bytes(cfg)


class TestTotal(unittest.TestCase):
    def test_total_scales_with_seq_and_batch(self):
        cfg = ModelConfig(
            layers=32, hidden_size=4096, num_heads=32, num_kv_heads=8, attention="gqa"
        )
        per = per_token_bytes(cfg)
        self.assertEqual(total_bytes(cfg, seq_len=100, batch=1), per * 100)
        self.assertEqual(total_bytes(cfg, seq_len=100, batch=3), per * 300)

    def test_sliding_window_caps_seq(self):
        cfg = ModelConfig(
            layers=4,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=4,
            attention="gqa",
            sliding_window=4096,
        )
        self.assertEqual(effective_seq_len(cfg, 100_000), 4096)
        self.assertEqual(
            total_bytes(cfg, seq_len=100_000), per_token_bytes(cfg) * 4096
        )

    def test_known_reference_value(self):
        # GQA: 32 層, hidden 4096, 32 Q heads, 8 KV heads, head_dim 128, fp16
        # per token = 32 layers * 2 * 8 * 128 * 2 bytes = 131072 bytes = 128 KiB
        # seq 8192 -> 1 GiB
        cfg = ModelConfig(
            layers=32, hidden_size=4096, num_heads=32, num_kv_heads=8, attention="gqa"
        )
        self.assertEqual(per_token_bytes(cfg), 131072)
        self.assertEqual(total_bytes(cfg, seq_len=8192), 1024**3)


class TestBudget(unittest.TestCase):
    def test_max_seq_len_for_budget(self):
        cfg = ModelConfig(
            layers=32, hidden_size=4096, num_heads=32, num_kv_heads=8, attention="gqa"
        )
        # 128 KiB/token なので 1 GiB では 8192 トークン
        self.assertEqual(max_seq_len_for_budget(cfg, 1024**3), 8192)

    def test_budget_respects_sliding_window(self):
        cfg = ModelConfig(
            layers=4,
            hidden_size=1024,
            num_heads=16,
            num_kv_heads=4,
            attention="gqa",
            sliding_window=1024,
        )
        self.assertEqual(max_seq_len_for_budget(cfg, 1024**4), 1024)


class TestFormat(unittest.TestCase):
    def test_format_bytes(self):
        self.assertEqual(format_bytes(512), "512 B")
        self.assertEqual(format_bytes(131072), "128.00 KiB")
        self.assertEqual(format_bytes(1024**3), "1.00 GiB")


if __name__ == "__main__":
    unittest.main()
