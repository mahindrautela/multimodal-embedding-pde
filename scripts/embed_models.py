#!/usr/bin/env python3
"""Frozen-model embedding backends used by run_embeddings.py.

Each backend exposes ``embed(items) -> np.ndarray [len(items), dim]`` where
an item is {"text": str | None, "image": PIL.Image | None}. Outputs are
L2-normalized float32. Input formatting follows each model card:

* All three run in bfloat16 on CPU (native AMX bf16 support); every output is
  re-normalized in float32.
* EmbeddingGemma 2 (sentence-transformers): text-bearing inputs use the
  documented symmetric prefix "task: sentence similarity | query: ";
  image-only inputs get no prefix; combined inputs place the deck text first
  and the <|image|> placeholder after it. Mean pooling + normalization are
  the checkpoint's own sentence-transformers modules.
* Qwen3-VL-Embedding-2B: official conversation format (system instruction
  "Represent the user's input.", user content = image then text, generation
  prompt appended), last-token pooling, L2 normalization; images resized by
  qwen-vl-utils with the official min/max pixel limits (factor 32).
* GME-Qwen2-VL-2B-Instruct: official gme_inference.py format (system
  "You are a helpful assistant.", image then text, "<|endoftext|>" appended),
  min_image_tokens=256, max_image_tokens=1280, max_length=1800, last-token
  pooling, L2 normalization.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch

MODEL_ROOT = Path(os.environ.get("PDE_MODEL_ROOT", "/home/claude/models"))
CHECKPOINTS = {
    "embeddinggemma2": ("google/embeddinggemma-2", "f9972ac22e696a105c482202646979e33e4b5a1e"),
    "qwen3vl2b": ("Qwen/Qwen3-VL-Embedding-2B", "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"),
    "gme2b": ("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", "9cfa6413f704a7c1cf5064d240748e10c876b286"),
}
GEMMA_PREFIX = "task: sentence similarity | query: "
QWEN3_INSTRUCTION = "Represent the user's input."
GME_INSTRUCTION = "You are a helpful assistant."


def snapshot(key: str) -> str:
    repo, revision = CHECKPOINTS[key]
    return str(MODEL_ROOT / ("models--" + repo.replace("/", "--")) / "snapshots" / revision)


def _renormalize(x) -> np.ndarray:
    """Re-normalize in float32 (bfloat16 outputs are only unit-norm to ~1e-3)."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class GemmaBackend:
    name = "embeddinggemma2"
    dtype_name = "bfloat16"

    def __init__(self, dtype=torch.bfloat16):
        from sentence_transformers import SentenceTransformer
        self.dtype_name = str(dtype).replace("torch.", "")
        self.model = SentenceTransformer(snapshot(self.name), device="cpu",
                                         model_kwargs={"torch_dtype": dtype},
                                         config_kwargs={"audio_config": None})
        self.model.eval()

    def describe(self) -> dict:
        return {"interface": "sentence-transformers", "text_prefix": GEMMA_PREFIX,
                "image_prefix": None, "combined_format": GEMMA_PREFIX + "{deck}\n<|image|>",
                "pooling": "checkpoint modules (mean pooling), then float32 L2 normalization", "dtype": self.dtype_name,
                "audio_encoder_loaded": False}

    def _format(self, item):
        text, image = item.get("text"), item.get("image")
        if text is not None and image is None:
            return GEMMA_PREFIX + text
        if text is None and image is not None:
            return {"image": image}
        return {"text": GEMMA_PREFIX + text + "\n<|image|>", "image": [image]}

    def token_count(self, item) -> int:
        proc = self.model[0].processor if hasattr(self.model[0], "processor") else None
        text, image = item.get("text"), item.get("image")
        tok = self.model.tokenizer if hasattr(self.model, "tokenizer") else proc.tokenizer
        n = len(tok(GEMMA_PREFIX + text)["input_ids"]) if text is not None else 0
        return n + (280 if image is not None else 0)

    @torch.inference_mode()
    def embed(self, items):
        inputs = [self._format(it) for it in items]
        out = self.model.encode(inputs, batch_size=len(inputs), normalize_embeddings=False,
                                convert_to_tensor=True, show_progress_bar=False)
        return _renormalize(out.float().cpu().numpy())


class _QwenFamilyBackend:
    """Shared last-token embedding code for Qwen3-VL and GME (Qwen2-VL)."""

    def _forward_last(self, inputs):
        out = self.model(**inputs, return_dict=True)
        hidden = out.last_hidden_state
        mask = inputs["attention_mask"]
        last = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
        emb = hidden[torch.arange(hidden.shape[0]), last].float()
        return _renormalize(emb.cpu().numpy())


class Qwen3VLBackend(_QwenFamilyBackend):
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
        conversations = [self._conversation(it) for it in items]
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images = None
        if any(it.get("image") is not None for it in items):
            images, _ = process_vision_info(conversations, image_patch_size=16)
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
        return self._forward_last(self.prepare(items))


class GMEBackend(_QwenFamilyBackend):
    name = "gme2b"
    dtype_name = "bfloat16"
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    max_length = 1800

    def __init__(self):
        from transformers import AutoProcessor, Qwen2VLModel
        path = snapshot(self.name)
        # PIL backend = the slow processor used by the reference code (transformers<4.52).
        self.processor = AutoProcessor.from_pretrained(path, min_pixels=self.min_pixels, max_pixels=self.max_pixels,
                                                       use_fast=False)
        self.processor.tokenizer.padding_side = "right"
        self.model = Qwen2VLModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval()

    def describe(self) -> dict:
        return {"interface": "transformers Qwen2VLModel (official gme_inference.py format)",
                "instruction": GME_INSTRUCTION, "content_order": "image, text",
                "min_pixels": self.min_pixels, "max_pixels": self.max_pixels, "resize_factor": 28,
                "max_length": self.max_length, "pooling": "last token (<|endoftext|>)",
                "dtype": self.dtype_name, "note": "card code uses float16 on GPU; bfloat16 used on CPU"}

    def _message(self, item):
        s = ""
        if item.get("image") is not None:
            s += "<|vision_start|><|image_pad|><|vision_end|>"
        if item.get("text") is not None:
            s += item["text"]
        return (f"<|im_start|>system\n{GME_INSTRUCTION}<|im_end|>\n<|im_start|>user\n{s}<|im_end|>\n"
                f"<|im_start|>assistant\n<|endoftext|>")

    def _fetch(self, image):
        """Exactly gme_inference.fetch_image: RGB + smart_resize(28, 4*28*28, 16384*28*28), PIL default resample."""
        from qwen_vl_utils.vision_process import smart_resize
        image = image.convert("RGB")
        h, w = smart_resize(image.height, image.width, factor=28, min_pixels=4 * 28 * 28,
                            max_pixels=16384 * 28 * 28)
        return image.resize((w, h))

    def processed_image_size(self, image):
        """Final (width, height) after fetch_image and the processor's own resize."""
        from qwen_vl_utils.vision_process import smart_resize
        im = self._fetch(image)
        h, w = smart_resize(im.height, im.width, factor=28, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        return (w, h)

    def prepare(self, items):
        texts = [self._message(it) for it in items]
        images = [self._fetch(it["image"]) for it in items if it.get("image") is not None]
        if images and len(images) != len(items):
            raise ValueError("GME batches must be all-image or all-text, as in the reference code")
        return self.processor(text=texts, images=images or None, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt")

    def token_count(self, item) -> int:
        texts = [self._message(item)]
        images = [self._fetch(item["image"])] if item.get("image") is not None else None
        enc = self.processor(text=texts, images=images, padding=False, truncation=False, return_tensors="pt")
        return int(enc["attention_mask"].sum())

    @torch.inference_mode()
    def embed(self, items):
        # The reference gme_inference.py (transformers<4.52) inserts the visual
        # features into the token embeddings and then calls the *text decoder*
        # with position_ids=None, i.e. plain sequential positions in all three
        # M-RoPE sections. Reproduce that exactly instead of the newer 3D index.
        inputs = self.prepare(items)
        ids = inputs["input_ids"]
        emb = self.model.get_input_embeddings()(ids)
        if "pixel_values" in inputs:
            feats = self.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"],
                                                  return_dict=True).pooler_output
            feats = torch.cat(list(feats), dim=0).to(emb.dtype)
            mask = (ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(emb)
            emb = emb.masked_scatter(mask, feats)
        positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, ids.shape[0], -1)
        out = self.model.language_model(inputs_embeds=emb, attention_mask=inputs["attention_mask"],
                                        position_ids=positions, return_dict=True)
        mask = inputs["attention_mask"]
        last = mask.sum(dim=1) - 1  # right padding, as in the reference code
        e = out.last_hidden_state[torch.arange(ids.shape[0]), last].float()
        return _renormalize(e.cpu().numpy())


BACKENDS = {"embeddinggemma2": GemmaBackend, "qwen3vl2b": Qwen3VLBackend, "gme2b": GMEBackend}


def load(name: str):
    torch.set_num_threads(int(os.environ.get("PDE_TORCH_THREADS", "2")))
    return BACKENDS[name]()
