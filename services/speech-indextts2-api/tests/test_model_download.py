import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from speech_runtime.models import download_file, read_sources


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class ModelDownload(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "model.bin"
        self.partial = self.path.with_name("model.bin.download")
        self.data = b"a fixed model object"
        self.item = {"path": "model.bin", "size": len(self.data), "sha256": hashlib.sha256(self.data).hexdigest(),
                     "url": "https://huggingface.co/example/model/resolve/pinned/model.bin"}

    def tearDown(self):
        self.directory.cleanup()

    def test_timeout_retries_then_resumes_interrupted_body(self):
        responses = [urllib.error.URLError("timeout"), Response(self.data[:5]),
                     Response(self.data[5:], 206, {"Content-Range": f"bytes 5-{len(self.data)-1}/{len(self.data)}"})]
        with patch("speech_runtime.models.urllib.request.urlopen", side_effect=responses) as request, patch("speech_runtime.models.time.sleep"):
            download_file(self.path, self.item, "https://hf-mirror.com")
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertEqual(request.call_args_list[2].args[0].get_header("Range"), "bytes=5-")
        self.assertFalse(self.partial.exists())

    def test_server_ignoring_range_replaces_partial_without_duplicate_bytes(self):
        self.partial.write_bytes(self.data[:5])
        with patch("speech_runtime.models.urllib.request.urlopen", return_value=Response(self.data)):
            download_file(self.path, self.item, "https://hf-mirror.com")
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_wrong_range_cannot_modify_partial_or_publish_model(self):
        self.partial.write_bytes(self.data[:5])
        with patch("speech_runtime.models.urllib.request.urlopen", return_value=Response(self.data[5:], 206, {"Content-Range": "bytes 0-1/2"})):
            with self.assertRaisesRegex(ValueError, "invalid_model_content_range"):
                download_file(self.path, self.item, "https://hf-mirror.com")
        self.assertEqual(self.partial.read_bytes(), self.data[:5])
        self.assertFalse(self.path.exists())

    def test_same_size_corruption_is_not_published(self):
        corrupt = b"x" * len(self.data)
        with patch("speech_runtime.models.urllib.request.urlopen", return_value=Response(corrupt)):
            with self.assertRaisesRegex(ValueError, "model_integrity_failed"):
                download_file(self.path, self.item, "https://hf-mirror.com")
        self.assertEqual(self.partial.read_bytes(), corrupt)
        self.assertFalse(self.path.exists())

    def test_complete_partial_is_verified_and_reused_without_network(self):
        self.partial.write_bytes(self.data)
        with patch("speech_runtime.models.urllib.request.urlopen") as request:
            download_file(self.path, self.item, "https://hf-mirror.com")
            request.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_retry_budget_preserves_progress_and_omits_remote_error_details(self):
        self.partial.write_bytes(self.data[:5])
        with patch("speech_runtime.models.urllib.request.urlopen", side_effect=urllib.error.URLError("private signed URL")) as request, patch("speech_runtime.models.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "^model_download_failed: model.bin$"):
                download_file(self.path, self.item, "https://hf-mirror.com", attempts=2)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(self.partial.read_bytes(), self.data[:5])
        self.assertFalse(self.path.exists())

    def test_mirror_source_preserves_locked_identity(self):
        source = dict(self.item, path="indextts2/model.bin", revision="a"*40)
        source["url"] = "https://www.modelscope.cn/models/IndexTeam/IndexTTS-2/resolve/" + source["revision"] + "/model.bin"
        manifest = {"files": [dict(self.item, path=source["path"])]}
        path = self.path.with_name("sources.json")
        path.write_text(json.dumps({"schema_version": 1, "files": [source]}))
        self.assertEqual(read_sources(manifest, path), {source["path"]: source["url"]})
        for change in ({"sha256": "0"*64}, {"revision": "master"}, {"url": "https://elsewhere.invalid/model.bin"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                path.write_text(json.dumps({"schema_version": 1, "files": [source | change]}))
                read_sources(manifest, path)

    def test_mirror_range_download_uses_exact_pinned_url(self):
        self.partial.write_bytes(self.data[:5])
        url = "https://www.modelscope.cn/models/IndexTeam/IndexTTS-2/resolve/" + "a"*40 + "/gpt.pth"
        with patch("speech_runtime.models.urllib.request.urlopen", return_value=Response(self.data[5:], 206, {"Content-Range": f"bytes 5-{len(self.data)-1}/{len(self.data)}"})) as request:
            download_file(self.path, self.item, "https://hf-mirror.com", source_url=url)
        self.assertEqual(request.call_args.args[0].full_url, url)
        self.assertEqual(self.path.read_bytes(), self.data)
