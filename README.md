# NPU PRC OpenVINO IR Model Packages

Private repository hosting OpenVINO IR model packages produced by NPU PRC
model conversion work. Each subfolder is a self-contained model package with
its own README.

## Packages

| Folder | Model | Notes |
|:-------|:------|:------|
| [`cosyvoice3-0.5b-2512/`](cosyvoice3-0.5b-2512/README.md) | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Zero-shot multilingual TTS (LLM + flow-matching + HiFTNet vocoder) |

Conversion scripts: https://github.com/yishanwang/cosyvoice3-openvino-conversion

## Git LFS

All `*.bin` and `*.onnx` files across every package are tracked via Git LFS
(see `.gitattributes`). Clone with `git lfs install` set up beforehand, or
run `git lfs pull` after cloning.

This is a **private** repository — not intended for public redistribution.
