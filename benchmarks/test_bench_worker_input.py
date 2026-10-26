import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bench_worker_input as bench


class Response(io.BytesIO):
    status = 200


class WorkerInputGuards(unittest.TestCase):
    def record(self):
        return {"Id": "a" * 64, "Image": "sha256:" + "b" * 64,
                "Config": {"Labels": {bench.OWNER_LABEL: "owner"},
                           "Cmd": ["--host", "127.0.0.1", "--port", "1234"], "Env": []},
                "HostConfig": {"NetworkMode": "host", "Runtime": "runc", "Privileged": False,
                               "DeviceRequests": None, "Devices": []}}

    def test_mutable_image_rejected(self):
        for image in ("latest", "repo:cuda", "sha256:1234", "sha256:" + "G" * 64):
            with self.assertRaises(ValueError):
                bench.immutable_image(image)
        self.assertEqual(bench.immutable_image("sha256:" + "f" * 64), "sha256:" + "f" * 64)

    def test_environment_isolated_and_pair_diff_only_ports(self):
        a = bench.environment(1234, 1236, "private-api-key", "private-engine-key")
        b = bench.environment(1235, 1236, "private-api-key", "private-engine-key")
        self.assertEqual({key for key in a if a[key] != b[key]}, {"PORT", "EZLOCALAI_PORT"})
        self.assertEqual(a["ROUTER_URL"], "")
        self.assertEqual(a["FALLBACK_SERVER"], "")
        self.assertEqual(a["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(a["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertEqual(a["GPU_LAYERS"], "0")
        for key in ("TTS_ENABLED", "STT_ENABLED", "IMAGE_ENABLED", "VIDEO_ENABLED", "MUSIC_ENABLED", "EMBEDDING_ENABLED"):
            self.assertEqual(a[key], "false")

    def test_command_loopback_no_gpu_and_key_only_environment(self):
        worker_environment = bench.environment(1234, 1235, "private-api-key", "private-engine-key")
        command = bench.docker_command("sha256:" + "b" * 64, Path("/worker"), Path("/data"),
                                       worker_environment, "name", "owner", 1234)
        self.assertEqual(command[-4:], ["--host", "127.0.0.1", "--port", "1234"])
        self.assertEqual(command[command.index("--network") + 1], "host")
        self.assertEqual(command[command.index("--runtime") + 1], "runc")
        environment_arguments = [command[index + 1] for index, value in enumerate(command) if value == "--env"]
        self.assertEqual(environment_arguments, list(worker_environment))
        self.assertTrue(all("=" not in argument for argument in environment_arguments))
        self.assertNotIn("private-api-key", " ".join(command))
        self.assertNotIn("private-engine-key", " ".join(command))
        self.assertNotIn("--env-file", command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--privileged", command)
        self.assertIn("type=bind,src=/worker,dst=/benchmark/worker,readonly", command)

    def test_environment_values_reach_only_the_docker_subprocess(self):
        worker_environment = {"EZLOCALAI_API_KEY": "private-api-key", "LLAMA_CPP_API_KEY": "private-engine-key"}
        original = dict(os.environ)
        with patch.object(bench.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            bench.docker(["run", "--env", "EZLOCALAI_API_KEY", "--env", "LLAMA_CPP_API_KEY"],
                         worker_environment=worker_environment)
        arguments = run.call_args.args[0]
        self.assertNotIn("private-api-key", arguments)
        self.assertNotIn("private-engine-key", arguments)
        for key, value in worker_environment.items():
            self.assertEqual(run.call_args.kwargs["env"][key], value)
        self.assertEqual(dict(os.environ), original)
        with self.assertRaises(ValueError):
            bench.docker_command("sha256:" + "b" * 64, Path("/worker"), Path("/data"),
                                 {"KEY=private": "value"}, "name", "owner", 1234)

    def test_main_keeps_credentials_out_of_files_arguments_and_output(self):
        api_key, engine_key = "api-private-do-not-store", "engine-private-do-not-store"
        image = "sha256:" + "b" * 64
        containers = {}
        removed = []
        temporary_directory = tempfile.TemporaryDirectory
        engine = SimpleNamespace(server=SimpleNamespace(server_port=3210), records=[])
        with temporary_directory() as root:
            folder = Path(root)
            binary = folder / "worker"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o700)

            def check_files():
                for path in folder.rglob("*"):
                    if path.is_file():
                        content = path.read_bytes()
                        self.assertNotIn(api_key.encode(), content)
                        self.assertNotIn(engine_key.encode(), content)

            def docker(arguments, required=True, worker_environment=None):
                check_files()
                self.assertNotIn(api_key, " ".join(arguments))
                self.assertNotIn(engine_key, " ".join(arguments))
                if arguments[0] == "run":
                    self.assertIsNotNone(worker_environment)
                    name = arguments[arguments.index("--name") + 1]
                    record = self.record()
                    record["Id"] = ("a" if not containers else "c") * 64
                    label = arguments[arguments.index("--label") + 1]
                    record["Config"]["Labels"] = {bench.OWNER_LABEL: label.split("=", 1)[1]}
                    record["Config"]["Env"] = [key + "=" + value for key, value in worker_environment.items()]
                    self.assertEqual(worker_environment["EZLOCALAI_API_KEY"], api_key)
                    self.assertEqual(worker_environment["LLAMA_CPP_API_KEY"], engine_key)
                    containers[name] = record
                    return SimpleNamespace(returncode=0, stdout=record["Id"])
                if arguments[0] == "inspect":
                    return SimpleNamespace(returncode=0, stdout=json.dumps([containers[arguments[1]]]))
                if arguments[0] == "rm":
                    removed.append(arguments[-1])
                    return SimpleNamespace(returncode=0)
                self.fail("unexpected Docker operation")

            args = ["bench_worker_input", "--image", image, "--baseline-binary", str(binary),
                    "--candidate-binary", str(binary), "--output", str(folder / "output"), "--blocks", "1"]
            output = io.StringIO()
            # argparse reads the process argv; all process/container calls are mocked.
            with patch("sys.argv", args), patch.object(bench, "docker", side_effect=docker), \
                 patch.object(bench, "MockEngine", return_value=contextlib.nullcontext(engine)), \
                 patch.object(bench, "cases", return_value={}), \
                 patch.object(bench.secrets, "token_hex", side_effect=[api_key, engine_key]), \
                 patch.object(bench, "open_loopback", side_effect=lambda *args, **kwargs: Response(b"{}")), \
                 patch.object(bench.tempfile, "TemporaryDirectory", side_effect=lambda **kwargs: temporary_directory(dir=root, **kwargs)), \
                 contextlib.redirect_stdout(output):
                bench.main()
            check_files()
            self.assertNotIn(api_key, output.getvalue())
            self.assertNotIn(engine_key, output.getvalue())
            self.assertEqual(len(removed), 2)
            self.assertTrue(json.loads((folder / "output" / "worker-input.json").read_text())["containers_removed"])

    def test_client_refuses_nonloopback_and_redirects(self):
        with patch.object(bench.DIRECT_OPENER, "open") as open_url:
            for url in ("http://example.com", "https://127.0.0.1", "http://private@127.0.0.1"):
                with self.assertRaises(ValueError):
                    bench.open_loopback(url, 1)
            open_url.assert_not_called()
        self.assertIsNone(bench.NoRedirect().redirect_request(None, None, None, None, None, None))

    def test_unowned_cleanup_refused_and_owned_removed_by_immutable_id(self):
        record = self.record()
        with patch.object(bench, "docker", return_value=SimpleNamespace(returncode=0, stdout=json.dumps([record]))) as run:
            with self.assertRaises(RuntimeError):
                bench.remove_owned("name", "another-owner")
            self.assertEqual(run.call_count, 1)
        with patch.object(bench, "docker", return_value=SimpleNamespace(returncode=0, stdout=json.dumps([record]))) as run:
            bench.remove_owned("name", "owner")
            self.assertEqual(run.call_args.args[0], ["rm", "--force", "a" * 64])

    def test_cleanup_attempts_all_owned_names_after_failure(self):
        with patch.object(bench, "remove_owned", side_effect=[RuntimeError("failed"), None]) as remove:
            with self.assertRaises(RuntimeError):
                bench.remove_all_owned(["a", "b"], "owner")
            self.assertEqual(remove.call_count, 2)

    def test_missing_container_is_distinct_from_daemon_error(self):
        for message in ("Error: No such object: gone\n", "error: no such object: gone\n"):
            absent = SimpleNamespace(returncode=1, stdout="[]", stderr=message)
            with patch.object(bench, "docker", return_value=absent) as run:
                bench.remove_owned("gone", "owner")
                self.assertEqual(run.call_count, 1)
        unavailable = SimpleNamespace(returncode=1, stdout="", stderr="Cannot connect to the Docker daemon")
        with patch.object(bench, "docker", return_value=unavailable):
            with self.assertRaises(RuntimeError):
                bench.remove_owned("name", "owner")

    def test_inspect_gpu_or_wrong_image_environment_rejected(self):
        record = self.record()
        bench.verify_isolation(record, "owner", record["Image"], {})
        with self.assertRaises(RuntimeError):
            bench.verify_isolation(record, "owner", "sha256:" + "c" * 64)
        with self.assertRaises(RuntimeError):
            bench.verify_isolation(record, "owner", expected_environment={"GPU_LAYERS": "0"})
        record["HostConfig"]["DeviceRequests"] = [{"Count": -1}]
        with self.assertRaises(RuntimeError):
            bench.verify_isolation(record, "owner")

    def test_workloads_have_no_media_side_effects_and_stable_payloads(self):
        workloads = bench.cases()
        self.assertEqual(len(workloads), 7)
        for name, messages in workloads.items():
            body = bench.payload(name, messages, True)
            self.assertEqual(body, bench.payload(name, messages, True))
            value = json.loads(body)
            self.assertTrue(value["stream"])
            self.assertNotIn("audio_format", value)
            self.assertNotIn("image_url", body.decode())
            self.assertNotIn("video_url", body.decode())

    def test_abba_counts_and_forwarded_identity(self):
        order = list(bench.blocks(6))
        self.assertEqual(order[:4], ["baseline", "candidate", "candidate", "baseline"])
        self.assertEqual(order.count("baseline"), 12)
        self.assertEqual(order.count("candidate"), 12)
        record = {"sha256": "abc", "bytes": 3, "path": "/v1/chat/completions"}
        self.assertEqual(bench.checked_capture([record], 0, record), record)
        with self.assertRaises(RuntimeError):
            bench.checked_capture([record], 0, {**record, "sha256": "different"})
        with self.assertRaises(RuntimeError):
            bench.checked_capture([record, record], 0, record)

    def test_buffered_identity_and_latency(self):
        body = json.dumps({"choices": [{"message": {"content": bench.ANSWER}, "finish_reason": "stop"}]}).encode()
        with patch.object(bench, "open_loopback", return_value=Response(body)):
            sample = bench.measure("http://127.0.0.1:1234", "private", b"{}", False,
                                   clock=iter([1.0, 1.1]).__next__)
        self.assertAlmostEqual(sample["full_ms"], 100)
        self.assertIsNone(sample["first_output_ms"])

    def test_stream_ttft_ignores_role_only_events_and_requires_done(self):
        values = [{"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                  {"choices": [{"delta": {"content": bench.ANSWER}, "finish_reason": None}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        body = b"".join(b"data: " + json.dumps(value).encode() + b"\n\n" for value in values)
        with patch.object(bench, "open_loopback", return_value=Response(body + b"data: [DONE]\n\n")):
            sample = bench.measure("http://127.0.0.1:1234", "private", b"{}", True,
                                   clock=iter([1.0, 1.04, 1.1]).__next__)
        self.assertAlmostEqual(sample["first_output_ms"], 40)
        with patch.object(bench, "open_loopback", return_value=Response(body)):
            with self.assertRaises(RuntimeError):
                bench.measure("http://127.0.0.1:1234", "private", b"{}", True)


if __name__ == "__main__":
    unittest.main()
