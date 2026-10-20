"""Offline Chinese ASR transcript verification using pinned OpenAI Whisper Small."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import time
import unicodedata

from .audio import digest_file

ASR_MODEL_ID = "whisper-small@9ecf779972d9"
ASR_FILES = {
    "small.pt": {
        "size": 483617219,
        "sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    },
    "multilingual.tiktoken": {
        "size": 816730,
        "sha256": "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126",
    },
    "mel_filters.npz": {
        "size": 4271,
        "sha256": "7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c",
    },
}

# Whisper special tokens + Simplified Chinese prompt ("以下是普通话的句子。")
WHISPER_EOT = 50257
WHISPER_NO_TIMESTAMPS = 50363
WHISPER_ZH_PROMPT_PREFIX = (
    50361,
    3588,
    4438,
    1541,
    29993,
    19550,
    21596,
    1546,
    34592,
    7626,
    1543,
    50258,
    50260,
    50359,
    50363,
)

_TRAD_TO_SIMP = str.maketrans({
    "著": "着", "裡": "里", "後": "后", "發": "发", "麵": "面", "兒": "儿", "風": "风",
    "緊": "紧", "繫": "系", "個": "个", "許": "许", "門": "门", "東": "东", "車": "车",
    "馬": "马", "鳥": "鸟", "魚": "鱼", "雲": "云", "氣": "气", "電": "电", "見": "见",
    "貝": "贝", "長": "长", "開": "开", "關": "关", "問": "问", "間": "间", "聞": "闻",
    "蘭": "兰", "謝": "谢", "張": "张", "劉": "刘", "陳": "陈", "楊": "杨", "趙": "赵",
    "黃": "黄", "吳": "吴", "孫": "孙", "羅": "罗", "鄭": "郑", "韓": "韩", "馮": "冯",
    "於": "于", "蕭": "萧", "鄧": "邓", "呂": "吕", "蘇": "苏", "盧": "卢", "蔣": "蒋",
    "賈": "贾", "葉": "叶", "閻": "阎", "餘": "余", "鐘": "钟", "範": "范", "譚": "谭",
    "鄒": "邹", "陸": "陆", "顧": "顾", "龍": "龙", "萬": "万", "錢": "钱", "湯": "汤",
    "喬": "乔", "賀": "贺", "賴": "赖", "龔": "龚", "賬": "账", "沒": "没", "別": "别",
    "頭": "头", "兩": "两", "這": "这", "誰": "谁", "麼": "么", "為": "为", "來": "来",
    "進": "进", "過": "过", "對": "对", "從": "从", "無": "无", "嗎": "吗", "讓": "让",
    "給": "给", "請": "请", "說": "说", "話": "话", "講": "讲", "讀": "读", "寫": "写",
    "聽": "听", "視": "视", "覺": "觉", "壞": "坏", "遠": "远", "舊": "旧", "輕": "轻",
    "難": "难", "熱": "热", "乾": "干", "濕": "湿", "淨": "净", "髒": "脏", "滿": "满",
    "動": "动", "靜": "静", "霧": "雾", "煙": "烟", "塵": "尘", "銀": "银", "銅": "铜",
    "鐵": "铁", "寶": "宝", "船": "船", "披": "披", "歲": "岁", "年": "年", "聲": "声",
    "音": "音", "響": "响", "語": "语", "言": "言", "詞": "词", "句": "句", "篇": "篇",
    "章": "章", "節": "节", "段": "段", "點": "点", "線": "线", "邊": "边", "角": "角",
    "處": "处", "所": "所", "場": "场", "院": "院", "樓": "楼", "層": "层", "室": "室",
    "廳": "厅", "堂": "堂", "廟": "庙", "橋": "桥", "路": "路", "街": "街", "巷": "巷",
    "村": "村", "鎮": "镇", "城": "城", "國": "国", "家": "家", "園": "园", "林": "林",
    "樹": "树", "花": "花", "草": "草", "木": "木", "葉": "叶", "根": "根", "枝": "枝",
    "條": "条", "塊": "块", "片": "片", "張": "张", "隻": "只", "雙": "双", "對": "对",
    "群": "群", "眾": "众", "隊": "队", "組": "组", "班": "班", "級": "级", "類": "类",
    "種": "种", "樣": "样", "式": "式", "型": "型", "號": "号", "碼": "码", "數": "数",
    "量": "量", "度": "度", "體": "体", "質": "质", "色": "色", "光": "光", "影": "影",
})

_DIGIT_MAP = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}


def verify_asr_models(root: Path | str) -> dict:
    root = Path(root)
    for name, spec in ASR_FILES.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != spec["size"] or digest_file(path) != spec["sha256"]:
            raise ValueError(f"asr_model_integrity_failed:{name}")
    return {"asr_model": ASR_MODEL_ID, "files": len(ASR_FILES)}


def _is_spoken_char(ch: str) -> bool:
    if not ch or ch.isspace():
        return False
    cat = unicodedata.category(ch)
    return cat.startswith(("L", "N"))


def _norm_char(ch: str) -> str:
    ch = ch.translate(_TRAD_TO_SIMP)
    if ch in _DIGIT_MAP:
        return _DIGIT_MAP[ch]
    return ch.lower()


def _compare_single(expected: str, asr_text: str) -> dict:
    exp_tokens = [(idx, ch, _norm_char(ch)) for idx, ch in enumerate(expected) if _is_spoken_char(ch)]
    act_tokens = [(idx, ch, _norm_char(ch)) for idx, ch in enumerate(asr_text) if _is_spoken_char(ch)]
    m, n = len(exp_tokens), len(act_tokens)
    if m == 0 and n == 0:
        return {"cer": 0.0, "differences": []}

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        exp_norm = exp_tokens[i - 1][2]
        for j in range(1, n + 1):
            act_norm = act_tokens[j - 1][2]
            cost = 0 if exp_norm == act_norm else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,  # match or substituted
                dp[i - 1][j] + 1,         # missing
                dp[i][j - 1] + 1,         # extra
            )

    diffs = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if exp_tokens[i - 1][2] == act_tokens[j - 1][2] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 1:
                    diffs.append({
                        "kind": "substituted",
                        "index": exp_tokens[i - 1][0],
                        "expected": exp_tokens[i - 1][1],
                        "heard": act_tokens[j - 1][1],
                    })
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            diffs.append({
                "kind": "missing",
                "index": exp_tokens[i - 1][0],
                "expected": exp_tokens[i - 1][1],
            })
            i -= 1
        else:
            anchor = exp_tokens[i][0] if i < m else (exp_tokens[-1][0] + 1 if m > 0 else 0)
            diffs.append({
                "kind": "extra",
                "index": anchor,
                "heard": act_tokens[j - 1][1],
            })
            j -= 1

    diffs.reverse()
    denom = max(1, m)
    cer = round(100.0 * len(diffs) / denom, 1) if diffs else 0.0
    return {"cer": cer, "differences": diffs}


def compare_transcript(expected: str, asr_text: str, source_text: str | None = None) -> dict:
    primary = _compare_single(expected, asr_text)
    if source_text and source_text != expected:
        alt = _compare_single(source_text, asr_text)
        if len(alt["differences"]) < len(primary["differences"]) or (
            len(alt["differences"]) == len(primary["differences"]) and alt["cer"] <= primary["cer"]
        ):
            return alt
    return primary


@dataclass
class _ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def _build_whisper_model(dims: _ModelDimensions):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LayerNorm(nn.LayerNorm):
        def forward(self, x):
            return F.layer_norm(
                x.float(),
                self.normalized_shape,
                None if self.weight is None else self.weight.float(),
                None if self.bias is None else self.bias.float(),
                self.eps,
            ).type(x.dtype)

    class Linear(nn.Linear):
        def forward(self, x):
            return F.linear(
                x,
                self.weight.to(x.dtype),
                None if self.bias is None else self.bias.to(x.dtype),
            )

    class Conv1d(nn.Conv1d):
        def _conv_forward(self, x, weight, bias):
            return super()._conv_forward(
                x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
            )

    def sinusoids(length: int, channels: int, max_timescale: int = 10000):
        log_inc = math.log(max_timescale) / (channels // 2 - 1)
        inv = torch.exp(-log_inc * torch.arange(channels // 2))
        scaled = torch.arange(length)[:, np.newaxis] * inv[np.newaxis, :]
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)

    class MultiHeadAttention(nn.Module):
        def __init__(self, n_state: int, n_head: int):
            super().__init__()
            self.n_head = n_head
            self.query = Linear(n_state, n_state)
            self.key = Linear(n_state, n_state, bias=False)
            self.value = Linear(n_state, n_state)
            self.out = Linear(n_state, n_state)

        def forward(self, x, xa=None, mask=None):
            q = self.query(x)
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
            n_batch, n_ctx, n_state = q.shape
            scale = (n_state // self.n_head) ** -0.25
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            w = F.softmax(qk.float(), dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            return self.out(out)

    class ResidualAttentionBlock(nn.Module):
        def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
            super().__init__()
            self.attn = MultiHeadAttention(n_state, n_head)
            self.attn_ln = LayerNorm(n_state)
            self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
            self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
            n_mlp = n_state * 4
            self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
            self.mlp_ln = LayerNorm(n_state)

        def forward(self, x, xa=None, mask=None):
            x = x + self.attn(self.attn_ln(x), mask=mask)
            if self.cross_attn is not None:
                x = x + self.cross_attn(self.cross_attn_ln(x), xa)
            x = x + self.mlp(self.mlp_ln(x))
            return x

    class AudioEncoder(nn.Module):
        def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
            super().__init__()
            self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
            self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
            self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
            self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
            self.ln_post = LayerNorm(n_state)

        def forward(self, x):
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)
            x = (x + self.positional_embedding).to(x.dtype)
            for block in self.blocks:
                x = block(x)
            return self.ln_post(x)

    class TextDecoder(nn.Module):
        def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
            super().__init__()
            self.token_embedding = nn.Embedding(n_vocab, n_state)
            self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
            self.blocks = nn.ModuleList(
                [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
            )
            self.ln = LayerNorm(n_state)
            mask = torch.empty(n_ctx, n_ctx).fill_(-math.inf).triu_(1)
            self.register_buffer("mask", mask, persistent=False)

        def forward(self, x, xa):
            x = self.token_embedding(x) + self.positional_embedding[: x.shape[-1]]
            x = x.to(xa.dtype)
            for block in self.blocks:
                x = block(x, xa, mask=self.mask)
            x = self.ln(x)
            return (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

    class WhisperNet(nn.Module):
        def __init__(self, d: _ModelDimensions):
            super().__init__()
            self.dims = d
            self.encoder = AudioEncoder(d.n_mels, d.n_audio_ctx, d.n_audio_state, d.n_audio_head, d.n_audio_layer)
            self.decoder = TextDecoder(d.n_vocab, d.n_text_ctx, d.n_text_state, d.n_text_head, d.n_text_layer)

    return WhisperNet(dims)


class WhisperSmallASR:
    """Offline Whisper Small Chinese ASR verifier sharing the runtime process."""

    def __init__(self, model_root: Path | str | None = None, device: str | None = None):
        if model_root is None:
            env_dir = os.environ.get("SPEECH_ASR_MODEL_ROOT")
            if env_dir:
                model_root = Path(env_dir)
            elif Path("/models-asr").is_dir():
                model_root = Path("/models-asr")
            else:
                model_root = Path("/models/whisper")
        self.root = Path(model_root)
        self.device = device
        self.model = None
        self.mel_filters = None
        self.vocab: dict[int, bytes] = {}
        self.ready = False
        self.error: str | None = "asr_model_not_loaded"
        self.load_seconds = 0.0

    def load(self) -> bool:
        started = time.monotonic()
        try:
            os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", "/tmp/torch-kernels")
            verify_asr_models(self.root)
            import numpy as np
            import torch

            device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self.device = device
            with np.load(self.root / "mel_filters.npz", allow_pickle=False) as f:
                self.mel_filters = torch.from_numpy(f["mel_80"]).to(device=device, dtype=torch.float32)

            vocab: dict[int, bytes] = {}
            for line in (self.root / "multilingual.tiktoken").read_bytes().splitlines():
                if line:
                    token_b64, rank = line.split()
                    vocab[int(rank)] = base64.b64decode(token_b64)
            self.vocab = vocab

            ckpt = torch.load(self.root / "small.pt", map_location="cpu")
            dims = _ModelDimensions(**ckpt["dims"])
            net = _build_whisper_model(dims)
            net.load_state_dict(ckpt["model_state_dict"], strict=True)
            net = net.to(device)
            if device.startswith("cuda"):
                net = net.half()
            net.eval()
            self.model = net
            self.ready = True
            self.error = None
            self.load_seconds = round(time.monotonic() - started, 4)
            return True
        except Exception as exc:
            self.model = None
            self.ready = False
            self.error = str(exc).split(":", 1)[0] if str(exc) else "asr_model_load_failed"
            return False

    def _transcribe_chunk(self, samples_16k) -> str:
        import torch
        import torch.nn.functional as F

        n_samples = 480000  # 30s at 16 kHz
        audio = torch.as_tensor(samples_16k, dtype=torch.float32, device=self.device)
        if audio.shape[-1] > n_samples:
            audio = audio[:n_samples]
        elif audio.shape[-1] < n_samples:
            audio = F.pad(audio, (0, n_samples - audio.shape[-1]))

        window = torch.hann_window(400, device=self.device)
        stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if self.device and self.device.startswith("cuda"):
            log_spec = log_spec.half()

        with torch.inference_mode():
            audio_features = self.model.encoder(log_spec.unsqueeze(0))
            tokens = list(WHISPER_ZH_PROMPT_PREFIX)
            generated: list[int] = []
            for _ in range(192):
                x = torch.tensor([tokens], dtype=torch.long, device=self.device)
                logits = self.model.decoder(x, audio_features)[0, -1]
                # Suppress timestamp tokens (> WHISPER_NO_TIMESTAMPS) and special control tokens except EOT
                logits[WHISPER_EOT + 1 :] = -math.inf
                next_token = int(torch.argmax(logits).item())
                if next_token == WHISPER_EOT:
                    break
                generated.append(next_token)
                tokens.append(next_token)
                if len(generated) >= 8 and len(set(generated[-8:])) == 1:
                    break

        raw_bytes = b"".join(self.vocab.get(t, b"") for t in generated if t < WHISPER_EOT)
        return raw_bytes.decode("utf-8", errors="replace").strip()

    def check_audio(self, wav_path: Path | str, expected_text: str, source_text: str | None = None) -> dict:
        if not self.ready or self.model is None:
            return {
                "transcript_check": {
                    "state": "unavailable",
                    "reason": self.error or "asr_model_not_loaded",
                },
                "asr_metrics": None,
            }
        started = time.monotonic()
        try:
            import librosa
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            samples, _ = librosa.load(str(wav_path), sr=16000, mono=True)
            chunk_size = 480000
            pieces = []
            for start in range(0, max(1, len(samples)), chunk_size):
                piece = samples[start : start + chunk_size]
                if len(piece) < 1600 and start > 0:
                    break
                text_piece = self._transcribe_chunk(piece)
                if text_piece:
                    pieces.append(text_piece)
            asr_text = "".join(pieces).strip()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_bytes = int(torch.cuda.max_memory_allocated())
            else:
                peak_bytes = 0
            elapsed = round(time.monotonic() - started, 4)
            diff = compare_transcript(expected_text, asr_text, source_text=source_text)
            return {
                "transcript_check": {
                    "state": "checked",
                    "asr_model": ASR_MODEL_ID,
                    "asr_text": asr_text,
                    "cer": diff["cer"],
                    "differences": diff["differences"],
                },
                "asr_metrics": {
                    "asr_model": ASR_MODEL_ID,
                    "elapsed_seconds": elapsed,
                    "peak_allocated_bytes": peak_bytes,
                },
            }
        except Exception:
            return {
                "transcript_check": {
                    "state": "unavailable",
                    "reason": "asr_inference_failed",
                },
                "asr_metrics": None,
            }
