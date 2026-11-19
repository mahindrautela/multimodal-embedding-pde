"""The three frozen embedding models, behind one small interface.

Each backend has
    embed(items)        -> float32 array [len(items), dim], unit length
    token_count(item)   -> number of input tokens (for the run records)
    describe()          -> the input-format settings stored with the embeddings
where an item is a dict {"text": str or None, "image": PIL image or None}.

All models run in bfloat16 on the CPU. bfloat16 outputs are only unit length
to about 1e-3, so every embedding is re-normalized in float32.

Input formats follow each model's own documentation:
* EmbeddingGemma 2 (sentence-transformers): text gets the prefix
  "task: sentence similarity | query: "; an image alone gets no prefix; a text
  + image input puts the text first and the <|image|> placeholder after it.
  Pooling and normalization are the checkpoint's own modules.
* Qwen3-VL-Embedding-2B: the official conversation format (system message
  "Represent the user's input.", then the image, then the text), last-token
  pooling. Images are resized by qwen-vl-utils with the official pixel limits.
* GME-Qwen2-VL-2B-Instruct: the format of the official gme_inference.py
  (system message "You are a helpful assistant.", image then text,
  "<|endoftext|>" at the end), 256-1280 image tokens, at most 1800 tokens,
  last-token pooling.

Model files are read from the Hugging Face cache layout under PDE_MODEL_ROOT
(default: HF_HUB_CACHE, else ~/.cache/huggingface/hub).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from common import CHECKPOINTS

MODEL_ROOT = Path(os.environ.get("PDE_MODEL_ROOT")
                  or os.environ.get("HF_HUB_CACHE")
                  or Path.home() / ".cache" / "huggingface" / "hub")
GEMMA_PREFIX = "task: sentence similarity | query: "
GEMMA_IMAGE_TOKENS = 280          # every image becomes 280 tokens in EmbeddingGemma 2
QWEN3_INSTRUCTION = "Represent the user's input."
GME_INSTRUCTION = "You are a helpful assistant."


def snapshot(key: str) -> str:
    """Local folder of the pinned checkpoint revision."""
    repo, revision = CHECKPOINTS[key]
    return str(MODEL_ROOT / ("models--" + repo.replace("/", "--")) / "snapshots" / revision)


def unit_rows(x) -> np.ndarray:
    """Convert to float32 and scale every row to unit length."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# EmbeddingGemma 2
# ---------------------------------------------------------------------------
class GemmaBackend:
    name = "embeddinggemma2"

    def __init__(self, dtype=torch.bfloat16):
        from sentence_transformers import SentenceTransformer
        self.dtype_name = str(dtype).replace("torch.", "")
        self.model = SentenceTransformer(snapshot(self.name), device="cpu",
                                         model_kwargs={"torch_dtype": dtype},
                                         config_kwargs={"audio_config": None})   # skip the audio encoder
        self.model.eval()

    def describe(self) -> dict:
        return {"interface": "sentence-transformers", "text_prefix": GEMMA_PREFIX,
                "image_prefix": None, "combined_format": GEMMA_PREFIX + "{deck}\n<|image|>",
                "pooling": "checkpoint modules (mean pooling), then float32 L2 normalization",
                "dtype": self.dtype_name, "audio_encoder_loaded": False}

    @staticmethod
    def _format(item):
        text, image = item.get("text"), item.get("image")
        if image is None:
            return GEMMA_PREFIX + text
        if text is None:
            return {"image": image}
        return {"text": GEMMA_PREFIX + text + "\n<|image|>", "image": [image]}

    def token_count(self, item) -> int:
        """Prefix + text tokens, plus the fixed image tokens (the placeholder line is not counted)."""
        text = item.get("text")
        n = len(self.model.tokenizer(GEMMA_PREFIX + text)["input_ids"]) if text is not None else 0
        return n + (GEMMA_IMAGE_TOKENS if item.get("image") is not None else 0)

    @torch.inference_mode()
    def embed(self, items):
        inputs = [self._format(item) for item in items]
        out = self.model.encode(inputs, batch_size=len(inputs), normalize_embeddings=False,
                                convert_to_tensor=True, show_progress_bar=False)
        return unit_rows(out.float().cpu().numpy())


# ---------------------------------------------------------------------------
# Qwen3-VL-Embedding-2B
# ---------------------------------------------------------------------------
class Qwen3VLBackend:
    name = "qwen3vl2b"
    dtype_name = "bfloat16"
    min_pixels = 4 * 32 * 32
    max_pixels = 1800 * 32 * 32
    max_length = 8192

    def __init__(self):
        from transformers import AutoProcessor, Qwen3VLModel
        path = snapshot(self.name)
        self.processor = AutoProcessor.from_pretrained(path)
        self.processor.tokenizer.padding_side = "right"
        self.model = Qwen3VLModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval()

    def describe(self) -> dict:
        return {"interface": "transformers Qwen3VLModel (official script format)",
                "instruction": QWEN3_INSTRUCTION, "content_order": "image, text",
                "min_pixels": self.min_pixels, "max_pixels": self.max_pixels, "resize_factor": 32,
                "max_length": self.max_length, "pooling": "last token", "dtype": self.dtype_name}

    def _conversation(self, item):
        content = []
        if item.get("image") is not None:
            content.append({"type": "image", "image": item["image"],
                            "min_pixels": self.min_pixels, "max_pixels": self.max_pixels})
        if item.get("text") is not None:
            content.append({"type": "text", "text": item["text"]})
        return [{"role": "system", "content": [{"type": "text", "text": QWEN3_INSTRUCTION}]},
                {"role": "user", "content": content}]

    def prepare(self, items):
        from qwen_vl_utils import process_vision_info
        conversations = [self._conversation(item) for item in items]
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images = None
        if any(item.get("image") is not None for item in items):
            images, _ = process_vision_info(conversations, image_patch_size=16)
        # Images are already resized by process_vision_info, hence do_resize=False.
        return self.processor(text=text, images=images, truncation=False, padding=True,
                              do_resize=False, return_tensors="pt")

    def token_count(self, item) -> int:
        return int(self.prepare([item])["attention_mask"].sum())

    def processed_image_size(self, image):
        from qwen_vl_utils import fetch_image
        return fetch_image({"image": image, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                           image_patch_size=16).size

    @torch.inference_mode()
    def embed(self, items):
        inputs = self.prepare(items)
        hidden = self.model(**inputs, return_dict=True).last_hidden_state
        mask = inputs["attention_mask"]
        last = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1     # last non-padding token
        return unit_rows(hidden[torch.arange(hidden.shape[0]), last].float().cpu().numpy())


# ---------------------------------------------------------------------------
# GME-Qwen2-VL-2B-Instruct
# ---------------------------------------------------------------------------
class GMEBackend:
    name = "gme2b"
    dtype_name = "bfloat16"
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    max_length = 1800

    def __init__(self):
        from transformers import AutoProcessor, Qwen2VLModel
        path = snapshot(self.name)
        # use_fast=False selects the PIL image processor used by the reference code.
        self.processor = AutoProcessor.from_pretrained(path, min_pixels=self.min_pixels,
                                                       max_pixels=self.max_pixels, use_fast=False)
        self.processor.tokenizer.padding_side = "right"
        self.model = Qwen2VLModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval()

    def describe(self) -> dict:
        return {"interface": "transformers Qwen2VLModel (official gme_inference.py format)",
                "instruction": GME_INSTRUCTION, "content_order": "image, text",
                "min_pixels": self.min_pixels, "max_pixels": self.max_pixels, "resize_factor": 28,
                "max_length": self.max_length, "pooling": "last token (<|endoftext|>)",
                "dtype": self.dtype_name, "note": "card code uses float16 on GPU; bfloat16 used on CPU"}

    @staticmethod
    def _message(item):
        content = ""
        if item.get("image") is not None:
            content += "<|vision_start|><|image_pad|><|vision_end|>"
        if item.get("text") is not None:
            content += item["text"]
        return (f"<|im_start|>system\n{GME_INSTRUCTION}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n"
                f"<|im_start|>assistant\n<|endoftext|>")

    @staticmethod
    def _fetch(image):
        """Same as gme_inference.fetch_image: RGB, then resize to multiples of 28 (PIL default filter)."""
        from qwen_vl_utils.vision_process import smart_resize
        image = image.convert("RGB")
        h, w = smart_resize(image.height, image.width, factor=28, min_pixels=4 * 28 * 28,
                            max_pixels=16384 * 28 * 28)
        return image.resize((w, h))

    def processed_image_size(self, image):
        """Final (width, height) after _fetch and the processor's own resize."""
        from qwen_vl_utils.vision_process import smart_resize
        fetched = self._fetch(image)
        h, w = smart_resize(fetched.height, fetched.width, factor=28,
                            min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        return (w, h)

    def prepare(self, items):
        texts = [self._message(item) for item in items]
        images = [self._fetch(item["image"]) for item in items if item.get("image") is not None]
        if images and len(images) != len(items):
            raise ValueError("GME batches must be all-image or all-text, as in the reference code")
        return self.processor(text=texts, images=images or None, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt")

    def token_count(self, item) -> int:
        images = [self._fetch(item["image"])] if item.get("image") is not None else None
        encoded = self.processor(text=[self._message(item)], images=images, padding=False,
                                 truncation=False, return_tensors="pt")
        return int(encoded["attention_mask"].sum())

    @torch.inference_mode()
    def embed(self, items):
        # The reference gme_inference.py (transformers < 4.52) puts the image
        # features into the token embeddings and then calls the text decoder
        # with position_ids=None, i.e. plain sequential positions in all three
        # rotary sections. We reproduce that instead of the newer 3D positions.
        inputs = self.prepare(items)
        ids = inputs["input_ids"]
        embeds = self.model.get_input_embeddings()(ids)
        if "pixel_values" in inputs:
            features = self.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"],
                                                     return_dict=True).pooler_output
            features = torch.cat(list(features), dim=0).to(embeds.dtype)
            image_slots = (ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(embeds)
            embeds = embeds.masked_scatter(image_slots, features)
        positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, ids.shape[0], -1)
        out = self.model.language_model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"],
                                        position_ids=positions, return_dict=True)
        last = inputs["attention_mask"].sum(dim=1) - 1                    # right padding
        return unit_rows(out.last_hidden_state[torch.arange(ids.shape[0]), last].float().cpu().numpy())


BACKENDS = {"embeddinggemma2": GemmaBackend, "qwen3vl2b": Qwen3VLBackend, "gme2b": GMEBackend}


def load(name: str, float32: bool = False):
    """Create a backend. Uses PDE_TORCH_THREADS CPU threads (default 2, as in the study).

    float32=True is a precision check that exists only for EmbeddingGemma 2.
    """
    torch.set_num_threads(int(os.environ.get("PDE_TORCH_THREADS", "2")))
    if float32:
        if name != "embeddinggemma2":
            raise ValueError("the float32 check is only available for embeddinggemma2")
        return GemmaBackend(dtype=torch.float32)
    return BACKENDS[name]()
