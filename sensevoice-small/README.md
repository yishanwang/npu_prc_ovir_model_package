# SenseVoiceSmall — OpenVINO IR

OpenVINO IR conversion of [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall),
a multilingual ASR / speech-emotion-recognition / audio-event-detection model.

Produced via the `funasr_onnx` ONNX export → `ovc` conversion path (SenseVoiceSmall
is a FunASR-native model, not a `transformers.AutoModel*`, so `optimum-cli export
openvino` does not apply). Conversion script (not included here — this repo is
the model package only): https://github.com/yishanwang/sensevoice-openvino-conversion

## Contents

| File | Role |
|:-----|:-----|
| `model.xml` / `.bin` | Encoder + CTC head, OpenVINO IR (FP16 weights) |
| `am.mvn` | Acoustic-feature mean/variance normalization stats (applied to mel features before inference) |
| `chn_jpn_yue_eng_ko_spectok.bpe.model` | Custom SentencePiece BPE tokenizer (decodes `ctc_logits` argmax IDs to text) |
| `config.yaml` | Original FunASR model hyperparameter config (reference only) |
| `configuration.json` | Original FunASR model-type metadata (confirms `"model": {"type": "funasr"}`, i.e. why this isn't an `optimum-intel` conversion) |

## Model architecture (from `config.yaml`)

| Component | Detail |
|:----------|:-------|
| Encoder | `SenseVoiceEncoderSmall` — SANM self-attention, 512 hidden, 4 heads, 2048 FFN, 50 blocks |
| Decoder | None — CTC-only, non-autoregressive |
| Frontend | 80-mel filterbank, 16 kHz, 25 ms/10 ms frame, LFR stacking (m=7, n=6) → 560-dim features |
| Tokenizer | Custom SentencePiece BPE (`chn_jpn_yue_eng_ko_spectok.bpe.model`) |

## Usage

This package alone is not a runnable pipeline — it needs a host-side
frontend (mel-spectrogram extraction, LFR stacking, `am.mvn` normalization)
and the SentencePiece tokenizer to decode CTC output. See the
[conversion-scripts repo](https://github.com/yishanwang/sensevoice-openvino-conversion)
for the export/convert steps and verified I/O signature.

## Verified I/O signature

```
Inputs:
  speech          [?, ?, 560]        f32   — LFR-stacked 80-mel features
  speech_lengths  [?]                i32
  language        [?]                i32   — LID selector
  textnorm        [?]                i32   — ITN on/off
Outputs:
  ctc_logits       [?, 4.., 25055]   f32   — CTC logits (text + LID/emotion/event tokens)
  encoder_out_lens [?]               i32
```

Confirmed with a real end-to-end run: ONNX export (2496 nodes, 935
initializers, ~897 MB weights) → `ovc` conversion → IR loaded and inspected
via `openvino.Core().read_model()`.

## License / Attribution

- Model weights derive from [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
  ([FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)). The
  original model's license terms apply.
- Tokenizer and normalization files derive unmodified from the same upstream
  repo.

This is a **private** repository — not intended for public redistribution.
