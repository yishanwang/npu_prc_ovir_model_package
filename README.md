# Fun-CosyVoice3-0.5B-2512 — OpenVINO IR

OpenVINO IR conversion of [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512),
a zero-shot multilingual TTS model (LLM + flow-matching + HiFTNet vocoder).

Produced with the official [`openvinotoolkit/openvino_notebooks` PR #3243](https://github.com/openvinotoolkit/openvino_notebooks/pull/3243)
reference conversion. Conversion scripts (not included here — this repo is
the model package only): https://github.com/yishanwang/cosyvoice3-openvino-conversion

## Contents

| File | Role |
|:-----|:-----|
| `openvino_model.xml` / `.bin` | Autoregressive LLM (Qwen2-0.5B backbone + speech-token head), **stateful** (KV-cache) |
| `openvino_text_embeddings_model.xml` / `.bin` | Qwen2 text-token embedding table |
| `openvino_speech_embeddings_model.xml` / `.bin` | CosyVoice speech-token embedding table |
| `openvino_flow_embeddings_model.xml` / `.bin` | Flow module: token embedding + pre-lookahead layer + speaker-embedding projection |
| `openvino_flow_estimator_model.xml` / `.bin` | Flow module: bare DiT flow-matching estimator (10-step Euler solver runs in host code, not in this graph) |
| `openvino_hift_model.xml` / `.bin` | HiFTNet vocoder, **up to but not including** the final ISTFT step (run `torch.istft` on the raw output in host code — see conversion-scripts repo for why) |
| `campplus.onnx` | Speaker embedding (pre-exported by FunAudioLLM, unmodified) |
| `speech_tokenizer_v3.onnx` | Speech tokenizer (pre-exported by FunAudioLLM, unmodified) |
| `cosyvoice3.yaml` | Original model hyperparameter config (reference only) |
| `CosyVoice-BlankEN/` | Qwen2 tokenizer files |

## Usage

This package alone is not a runnable pipeline — it needs the CosyVoice
Python package (tokenization, mel-spectrogram extraction, sampling loop,
ISTFT post-processing) to orchestrate inference across these sub-models.
See the `OVCosyVoice3`/`OVCosyVoice3LM`/`OVFlow`/`OVHiFT` wrapper classes in
`ov_cosyvoice_helper.py` (vendored in the
[conversion-scripts repo](https://github.com/yishanwang/cosyvoice3-openvino-conversion)'s
`third_party/`) for a working inference pipeline built on these IR files.

## Validation

- LLM: validated with a real autoregressive `infer_request.infer()` loop
  (via `OVCosyVoice3LM`) — produced valid speech tokens end-to-end.
- Flow: validated `ov.convert_model()` + `compile_model()` + `infer()` output
  against the PyTorch reference (~0.06 max abs diff, expected float32 drift
  across a 22-layer DiT).
- HiFT (this package's variant, from the official helper, no-ISTFT): the
  neural-network portion is validated the same way; the separate ISTFT
  post-processing step reuses PyTorch's own `torch.istft` unmodified so it
  carries no additional conversion risk.

## License / Attribution

- Model weights derive from [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
  ([FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice), Apache
  License 2.0). This package is a derived/converted form of those weights;
  the original model's license terms apply.
- `CosyVoice-BlankEN/` tokenizer files derive from the Qwen2 tokenizer
  (Apache License 2.0 / per Qwen's own license terms).
- Conversion performed using the vendored, unmodified
  `openvinotoolkit/openvino_notebooks` PR #3243 helper (Apache License 2.0).

This is a **private** repository — not intended for public redistribution.
