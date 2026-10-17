import importlib.util
from pathlib import Path
import shutil
import stat
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("partition_environment", Path(__file__).parents[1] / "scripts" / "partition_environment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EnvironmentLayersTests(unittest.TestCase):
    def test_reconstruction_preserves_files_links_and_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "environment"
            (source / "bin").mkdir(parents=True)
            (source / "lib" / "empty").mkdir(parents=True)
            (source / "bin" / "tool").write_bytes(b"executable")
            (source / "bin" / "tool").chmod(0o755)
            (source / "lib" / "large.so").write_bytes(b"large" * 20)
            (source / "lib" / "small.so").write_bytes(b"small")
            (source / "bin" / "python").symlink_to("/usr/local/bin/python3.11")
            (source / "lib64").symlink_to("lib", target_is_directory=True)
            report = module.partition(source, root / "layers", 3)
            self.assertEqual(report["files"], 5)
            reconstructed = root / "restored"
            for layer in sorted((root / "layers").iterdir()):
                shutil.copytree(layer, reconstructed, dirs_exist_ok=True, symlinks=True)
            self.assertEqual((reconstructed / "lib" / "large.so").read_bytes(), b"large" * 20)
            self.assertEqual((reconstructed / "lib64" / "small.so").read_bytes(), b"small")
            self.assertEqual((reconstructed / "bin" / "python").readlink().as_posix(), "/usr/local/bin/python3.11")
            self.assertEqual(stat.S_IMODE((reconstructed / "bin" / "tool").stat().st_mode), 0o755)
            self.assertTrue((reconstructed / "lib" / "empty").is_dir())
            with self.assertRaises(ValueError):
                module.partition(source, root / "layers")


if __name__ == "__main__":
    unittest.main()
