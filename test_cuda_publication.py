import json
import subprocess
import tempfile
import unittest
from pathlib import Path

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

    def runner(self, command, **kwargs):
        self.calls.append(command)
        if command[1] == self.fail:
            raise subprocess.CalledProcessError(1, command)
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
        self.assertEqual(self.calls[1], ["docker", "push", IMAGE])
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


if __name__ == "__main__":
    unittest.main()
