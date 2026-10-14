"""Fetch the locked upstream source and apply count-checked offline-loading patches."""
import hashlib
import json
import os
import pathlib
import sys
import tarfile
import tempfile
import urllib.request


def replace_once(root, file, before, after):
    path = root / file
    text = path.read_text()
    if text.count(before) != 1:
        raise RuntimeError('upstream_patch_mismatch: ' + file)
    path.write_text(text.replace(before, after))


def prepare(lock_path, destination, archive_path=None):
    lock = json.loads(pathlib.Path(lock_path).read_text())
    destination = pathlib.Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        archive = pathlib.Path(archive_path) if archive_path else pathlib.Path(temporary) / 'source.tar.gz'
        if not archive_path:
            with urllib.request.urlopen(lock['url'], timeout=120) as source, archive.open('wb') as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
        if archive.stat().st_size != lock['size'] or hashlib.sha256(archive.read_bytes()).hexdigest() != lock['sha256']:
            raise RuntimeError('upstream_archive_integrity_failed')
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                parts = pathlib.PurePosixPath(member.name).parts
                if member.issym() or member.islnk() or '..' in parts or member.name.startswith('/'):
                    raise RuntimeError('unsafe_source_archive')
            source.extractall(temporary, filter='data')
        extracted = pathlib.Path(temporary) / ('index-tts-' + lock['revision'])
        for entry in extracted.iterdir():
            entry.rename(destination / entry.name)
    patches = [
        ('indextts/infer_v2.py', "os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'", "os.environ.setdefault('HF_HUB_CACHE', '/tmp/huggingface')"),
        ('indextts/infer_v2.py', 'SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")', 'SeamlessM4TFeatureExtractor.from_pretrained(os.path.join(os.environ["SPEECH_MODEL_ROOT"], "w2v-bert"), local_files_only=True)'),
        ('indextts/infer_v2.py', 'hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")', 'os.path.join(os.environ["SPEECH_MODEL_ROOT"], "maskgct/semantic_codec/model.safetensors")'),
        ('indextts/infer_v2.py', 'hf_hub_download(\n            "funasr/campplus", filename="campplus_cn_common.bin"\n        )', 'os.path.join(os.environ["SPEECH_MODEL_ROOT"], "campplus/campplus_cn_common.bin")'),
        ('indextts/infer_v2.py', 'bigvgan_name = self.cfg.vocoder.name', 'bigvgan_name = os.path.join(os.environ["SPEECH_MODEL_ROOT"], "bigvgan")'),
        ('indextts/utils/maskgct_utils.py', 'import torch\n', 'import os\nimport torch\n'),
        ('indextts/utils/maskgct_utils.py', 'Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")', 'Wav2Vec2BertModel.from_pretrained(os.path.join(os.environ["SPEECH_MODEL_ROOT"], "w2v-bert"), local_files_only=True)'),
        ('indextts/utils/front.py', 'NormalizerEn(overwrite_cache=False)', 'NormalizerEn(cache_dir=cache_dir, overwrite_cache=False)'),
    ]
    for patch in patches:
        replace_once(destination, *patch)
    print(json.dumps({'revision': lock['revision'], 'patches': len(patches)}))


if __name__ == '__main__':
    prepare(*sys.argv[1:])
