"""Boundary contracts for the public calculation API."""

import unittest
from dataclasses import replace

from kv_cache_sim.core import (
    ModelConfig, bytes_per_element, effective_seq_len, format_bytes,
    max_seq_len_for_budget, per_token_bytes, per_token_bytes_per_layer, total_bytes,
)


class TestBoundaries(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig(layers=2, hidden_size=64, num_heads=4,
                               num_kv_heads=2)

    def test_nonpositive_sequence_lengths(self):
        for window in (None, 1, 4):
            cfg = replace(self.cfg, sliding_window=window)
            for seq in (0, -1, -1024):
                with self.subTest(window=window, seq=seq):
                    with self.assertRaises(ValueError):
                        effective_seq_len(cfg, seq)
                    with self.assertRaises(ValueError):
                        total_bytes(cfg, seq)

    def test_nonpositive_budgets(self):
        for budget in (0, -1, -1024**3):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                max_seq_len_for_budget(self.cfg, budget)

    def test_budget_token_boundary(self):
        # 2 layers * 2 K/V * 2 KV heads * 16 dimensions * 2 bytes.
        for batch in (1, 3):
            cost = 256 * batch
            for budget, expected in ((1, 0), (cost - 1, 0), (cost, 1),
                                     (cost * 2 - 1, 1), (cost * 2, 2)):
                with self.subTest(batch=batch, budget=budget):
                    self.assertEqual(max_seq_len_for_budget(
                        self.cfg, budget, batch=batch), expected)

    def test_huge_budget(self):
        # A power of two stays exact in the existing floating point arithmetic.
        self.assertEqual(max_seq_len_for_budget(self.cfg, 2**100), 2**92)
        self.assertEqual(max_seq_len_for_budget(
            replace(self.cfg, sliding_window=4), 2**100), 4)

    def test_extreme_and_fractional_byte_precisions(self):
        for bits, bpe, layer, token in ((1, 0.125, 8, 16),
                                       (12, 1.5, 96, 192),
                                       (128, 16, 1024, 2048)):
            with self.subTest(bits=bits):
                self.assertEqual(bytes_per_element(bits), bpe)
                self.assertEqual(per_token_bytes_per_layer(self.cfg, bits), layer)
                self.assertEqual(per_token_bytes(self.cfg, bits), token)
                self.assertEqual(total_bytes(self.cfg, 3, batch=2, bits=bits), token * 6)
                self.assertEqual(max_seq_len_for_budget(
                    self.cfg, token * 6, batch=2, bits=bits), 3)

    def test_nonpositive_bits(self):
        for bits in (0, -1, -128):
            calls = (
                lambda: bytes_per_element(bits),
                lambda: per_token_bytes_per_layer(self.cfg, bits),
                lambda: per_token_bytes(self.cfg, bits),
                lambda: total_bytes(self.cfg, 1, bits=bits),
                lambda: max_seq_len_for_budget(self.cfg, 1024, bits=bits),
            )
            for index, call in enumerate(calls):
                with self.subTest(bits=bits, api=index), self.assertRaises(ValueError):
                    call()

    def test_nonpositive_batch(self):
        for batch in (0, -1, -100):
            with self.subTest(batch=batch):
                with self.assertRaises(ValueError):
                    total_bytes(self.cfg, 1, batch=batch)
                with self.assertRaises(ValueError):
                    max_seq_len_for_budget(self.cfg, 1024, batch=batch)

    def test_sliding_window_boundaries(self):
        for window in (1, 4):
            cfg = replace(self.cfg, sliding_window=window)
            for seq in (1, window, window + 1):
                expected = min(seq, window)
                with self.subTest(window=window, seq=seq):
                    self.assertEqual(effective_seq_len(cfg, seq), expected)
                    self.assertEqual(total_bytes(cfg, seq, batch=2), 256 * expected * 2)
            self.assertEqual(max_seq_len_for_budget(cfg, 256 * window), window)
            self.assertEqual(max_seq_len_for_budget(cfg, 256 * (window + 1)), window)

    def test_invalid_configuration_values(self):
        cases = [
            {field: value}
            for field in ('layers', 'num_heads', 'num_kv_heads', 'sliding_window')
            for value in (0, -1)
        ] + [
            {'num_kv_heads': 5}, {'num_kv_heads': 3},
            {'hidden_size': 65}, {'attention': 'unknown'},
            *({'attention': 'mla', 'kv_lora_rank': rank} for rank in (None, 0, -1)),
        ]
        for changes in cases:
            cfg = replace(self.cfg, **changes)
            for name, call in (
                ('validate', cfg.validate),
                ('per_token', lambda: per_token_bytes(cfg)),
                ('total', lambda: total_bytes(cfg, 1)),
                ('budget', lambda: max_seq_len_for_budget(cfg, 1024)),
            ):
                with self.subTest(changes=changes, api=name), self.assertRaises(ValueError):
                    call()

    def test_format_byte_boundaries(self):
        for value, expected in ((0, '0 B'), (1, '1 B'), (1023, '1023 B'),
                                (1024, '1.00 KiB'), (1024**5, '1.00 PiB'),
                                (1024**8, '1073741824.00 PiB')):
            with self.subTest(value=value):
                self.assertEqual(format_bytes(value), expected)
