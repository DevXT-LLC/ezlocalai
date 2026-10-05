from pathlib import Path
import stat
import tempfile
import unittest

from scripts.configure_gb10_pools import configure_pools


class Gb10PoolConfigurationTests(unittest.TestCase):
    def test_updates_only_pool_sizes_and_preserves_private_original_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            original = "# Worker settings\nSECRET=unchanged\nTTS_N_PARALLEL=4\nVIDEO_ENABLED=true\n"
            path.write_text(original)
            configure_pools(path, 2)
            expected = "# Worker settings\nSECRET=unchanged\nTTS_N_PARALLEL=2\nVIDEO_ENABLED=true\nSTT_N_PARALLEL=2\nEMBEDDING_N_PARALLEL=2\n"
            self.assertEqual(path.read_text(), expected)
            backup = path.with_name(".env.pre-gb10-pools")
            self.assertEqual(backup.read_text(), original)
            self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            configure_pools(path, 1)
            self.assertEqual(backup.read_text(), original)
            self.assertIn("TTS_N_PARALLEL=1\n", path.read_text())

    def test_invalid_count_does_not_change_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("TTS_N_PARALLEL=4\n")
            with self.assertRaises(ValueError):
                configure_pools(path, 0)
            self.assertEqual(path.read_text(), "TTS_N_PARALLEL=4\n")
            self.assertEqual(list(path.parent.iterdir()), [path])
