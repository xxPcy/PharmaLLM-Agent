"""CPU-only checks for accidental publication and the exported environment bootstrap."""

import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
REPO = TOOLS.parent
spec = importlib.util.spec_from_file_location("public_check", TOOLS / "check_public_repo.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class PublicRepoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "README.md").write_text("Public fixture\n")

    def reasons(self, result):
        return {reason for finding in result["findings"] for reason in finding["reasons"]}

    def test_clean_tree(self):
        self.assertTrue(check.check(self.root)["ok"])

    def test_real_environment_rejected(self):
        (self.root / ".env").write_text("not a public file")
        self.assertIn("private_environment", self.reasons(check.check(self.root)))

    def test_example_environment_allowed(self):
        (self.root / ".env.example").write_text("JWT_SECRET=CHANGE_ME\n")
        self.assertTrue(check.check(self.root)["ok"])

    def test_weight_and_image_rejected(self):
        (self.root / "weights.pth").write_bytes(b"fixture")
        (self.root / "case.nii.gz").write_bytes(b"fixture")
        self.assertEqual(2, len(check.check(self.root)["findings"]))

    def test_data_implementation_kept(self):
        path = self.root / "PharmaLLM_V1.0/src/pharma_llm/data/schema.py"
        path.parent.mkdir(parents=True)
        path.write_text("# implementation\n")
        self.assertTrue(check.check(self.root)["ok"])

    def test_dataset_rejected(self):
        (self.root / "training.jsonl").write_text("{}\n")
        self.assertIn("unreviewed_dataset", self.reasons(check.check(self.root)))

    def test_symlink_not_followed(self):
        (self.root / "linked.txt").symlink_to("README.md")
        self.assertIn("symlink_not_allowed", self.reasons(check.check(self.root)))

    def test_personal_path_rejected(self):
        (self.root / "config.txt").write_text("/" + "home/private-user/model")
        self.assertIn("personal_home_path", self.reasons(check.check(self.root)))

    def test_known_secret_only_reports_path(self):
        secret = b"synthetic-secret-for-comparison"
        (self.root / "config.txt").write_bytes(secret)
        result = check.check(self.root, secrets=[secret])
        self.assertIn("matches_local_secret", self.reasons(result))
        self.assertNotIn(secret.decode(), json.dumps(result))

    def test_token_only_reports_path(self):
        token = b"gh" + b"p_" + b"x" * 36
        (self.root / "config.txt").write_bytes(token)
        result = check.check(self.root)
        self.assertIn("github_token", self.reasons(result))
        self.assertNotIn(token.decode(), json.dumps(result))

    def test_size_limits(self):
        with patch.object(check, "MAX_FILE_BYTES", 2):
            self.assertIn("file_too_large", self.reasons(check.check(self.root)))
        with patch.object(check, "MAX_TOTAL_BYTES", 2):
            self.assertIn("snapshot_too_large", self.reasons(check.check(self.root)))

    def test_staged_content_checked_after_working_copy_cleaned(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        path = self.root / "config.txt"
        path.write_bytes(b"gh" + b"p_" + b"z" * 36)
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True, capture_output=True)
        path.write_text("clean working copy\n")
        self.assertTrue(check.check(self.root)["ok"])
        self.assertIn("github_token", self.reasons(check.check(self.root, staged=True)))

    def test_staged_symlink_rejected(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        (self.root / "link").symlink_to("README.md")
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True, capture_output=True)
        self.assertIn("non_regular_git_entry", self.reasons(check.check(self.root, staged=True)))

    def test_private_env_comparison_without_display(self):
        private = self.root / "private-config.txt"
        private.write_text('JWT_SECRET="synthetic-local-only-secret-value"\nMODEL_MODE=fake\n')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            values = check.private_values(private)
        self.assertEqual([b"synthetic-local-only-secret-value"], values)
        self.assertEqual("", output.getvalue())

    def test_environment_bootstrap_generates_distinct_private_secrets(self):
        app = self.root / "app"
        scripts = app / "scripts"
        scripts.mkdir(parents=True)
        original_app = REPO / "PharmaLLM_V1.1/agent_app"
        if not original_app.is_dir():
            self.skipTest("Only applicable to the exported repository")
        shutil.copy2(original_app / ".env.example", app / ".env.example")
        shutil.copy2(original_app / "scripts/init_env.py", scripts / "init_env.py")
        subprocess.run([sys.executable, str(scripts / "init_env.py"), "--mode", "fake"],
                       check=True, capture_output=True)
        private = app / ".env"
        self.assertEqual(0o600, private.stat().st_mode & 0o777)
        values = dict(line.split("=", 1) for line in private.read_text().splitlines()
                      if line and not line.startswith("#") and "=" in line)
        tokens = [values[key] for key in ("JWT_SECRET", "POSTGRES_PASSWORD", "SEGMENTATION_API_KEY")]
        self.assertTrue(all(len(token) >= 32 and token != "CHANGE_ME" for token in tokens))
        self.assertEqual(3, len(set(tokens)))
        self.assertEqual("fake", values["MODEL_MODE"])
        before = private.read_bytes()
        result = subprocess.run([sys.executable, str(scripts / "init_env.py")], capture_output=True, check=False)
        self.assertNotEqual(0, result.returncode)
        self.assertEqual(before, private.read_bytes())


if __name__ == "__main__":
    unittest.main()
