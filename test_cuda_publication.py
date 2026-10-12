import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import publish_cuda_image as publisher

OLD = "sha256:" + "a" * 64
NEW = "sha256:" + "b" * 64
IMAGE = "localhost:5000/devxt/ezlocalai-cuda:latest"


class FakeRegistry:
    def __init__(self):
        self.latest = OLD
        self.deleted = []
        self.fail_delete = False

    def digest(self, reference):
        return self.latest

    def tagged_digests(self):
        return {OLD}

    def delete(self, digest):
        if self.fail_delete:
            raise RuntimeError("cleanup failed")
        self.deleted.append(digest)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.registry = FakeRegistry()
        self.calls = []
        self.fail = None
        self.pushed_digest = NEW
        self.build_platform = "linux/amd64"

    def runner(self, command, **kwargs):
        self.calls.append(command)
        if command[1] == self.fail:
            raise subprocess.CalledProcessError(1, command)
        if command[1] == "image":
            return subprocess.CompletedProcess(
                command, 0, self.build_platform + "\n", ""
            )
        if command[1] == "push":
            self.registry.latest = self.pushed_digest
            return subprocess.CompletedProcess(
                command, 0, f"latest: digest: {self.pushed_digest} size: 100\n", ""
            )
        return subprocess.CompletedProcess(command, 0)

    def publish(self):
        publisher.publish(IMAGE, self.registry, self.state, self.runner)

    def test_success_publishes_only_latest_and_deletes_previous_manifest(self):
        self.publish()
        self.assertEqual(self.registry.deleted, [OLD])
        self.assertEqual(self.registry.latest, NEW)
        self.assertEqual(self.calls[2], ["docker", "push", IMAGE])
        self.assertEqual(json.loads((self.state / "pending.json").read_text()), [])

    def test_build_and_push_failures_preserve_old_manifest(self):
        for command in ["build", "push"]:
            self.fail = command
            with self.assertRaises(subprocess.CalledProcessError):
                self.publish()
            self.assertEqual(self.registry.deleted, [])
            self.assertEqual(self.registry.latest, OLD)

    def test_no_op_build_never_deletes_latest_digest(self):
        self.pushed_digest = OLD
        self.publish()
        self.assertEqual(self.registry.deleted, [])

    def test_digest_mismatch_does_not_delete_anything(self):
        def mismatched_runner(command, **kwargs):
            result = self.runner(command, **kwargs)
            if command[1] == "push":
                self.registry.latest = OLD
            return result

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            publisher.publish(IMAGE, self.registry, self.state, mismatched_runner)
        self.assertEqual(self.registry.deleted, [])

    def test_failed_cleanup_is_retried_even_after_old_latest_is_untagged(self):
        self.registry.fail_delete = True
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.publish()
        self.assertEqual(json.loads((self.state / "pending.json").read_text()), [OLD])
        self.registry.fail_delete = False
        self.registry.tagged_digests = lambda: {NEW}
        self.publish()
        self.assertEqual(self.registry.deleted, [OLD])

    def test_requires_latest_tag_and_registry(self):
        for image in ["ezlocalai:latest", "localhost:5000/devxt/ezlocalai-cuda:old"]:
            with self.assertRaises(ValueError):
                publisher.Registry(image)

    def test_gb10_builds_and_verifies_arm64_before_push(self):
        self.build_platform = "linux/arm64"
        image = "localhost:5000/devxt/ezlocalai-gb10:latest"
        publisher.publish(image, self.registry, self.state, self.runner, variant="gb10")
        self.assertEqual(
            self.calls[0],
            [
                "docker",
                "build",
                "--platform",
                "linux/arm64",
                "--file",
                "gb10.Dockerfile",
                "--tag",
                image,
                ".",
            ],
        )
        self.assertEqual(self.calls[2], ["docker", "push", image])
        self.assertEqual(self.registry.deleted, [OLD])

    def test_wrong_architecture_does_not_push_or_delete_previous_image(self):
        with self.assertRaisesRegex(RuntimeError, "expected linux/arm64"):
            publisher.publish(
                IMAGE, self.registry, self.state, self.runner, variant="gb10"
            )
        self.assertEqual(self.registry.deleted, [])
        self.assertFalse(any(command[1] == "push" for command in self.calls))

    def test_cli_separates_cuda_and_gb10_retention_state(self):
        states = []
        images = []
        for variant in ["cuda", "gb10"]:
            with patch(
                "sys.argv", ["publish_cuda_image.py", "--variant", variant]
            ), patch.object(publisher, "publish") as publish, patch.dict(
                publisher.os.environ,
                {
                    "EZLOCALAI_CUDA_IMAGE": IMAGE,
                    "EZLOCALAI_GB10_IMAGE": "localhost:5000/devxt/ezlocalai-gb10:latest",
                },
            ):
                publisher.main()
            images.append(publish.call_args.args[0])
            states.append(publish.call_args.args[2])
        self.assertNotEqual(images[0], images[1])
        self.assertNotEqual(states[0], states[1])
        self.assertEqual(states[0], Path.home() / ".local/state/ezlocalai-cuda-publish")


if __name__ == "__main__":
    unittest.main()
