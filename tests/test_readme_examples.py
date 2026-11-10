"""Run the commands documented in README and verify their published output."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestReadmeExamples(unittest.TestCase):
    def check_example(self, index, command_kind):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        examples = re.findall(r'```console\n(.*?)\n```', readme, re.DOTALL)
        self.assertGreaterEqual(len(examples), 3)
        lines = examples[index].splitlines()
        command = lines.pop(0).removeprefix('$ ')
        while command.endswith('\\'):
            command = command[:-1] + lines.pop(0).strip()
        argv = shlex.split(command)
        self.assertEqual(argv[:4], ['PYTHONPATH=src', 'python3', '-m', 'kv_cache_sim'])
        self.assertEqual(argv[4], command_kind)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'src')
        result = subprocess.run(
            [sys.executable, *argv[2:]], cwd=ROOT, env=env,
            capture_output=True, text=True, encoding='utf-8', timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        # Compare every documented line, ignoring only alignment whitespace.
        self.assertEqual(
            [' '.join(line.split()) for line in result.stdout.splitlines()],
            [' '.join(line.split()) for line in lines],
        )

    def test_gqa_calc_example(self):
        self.check_example(0, 'calc')

    def test_mla_calc_example(self):
        self.check_example(1, 'calc')

    def test_compare_table(self):
        self.check_example(2, 'compare')
