# Qualification kaggle-t4-20261025-c

Baseline: pinned Qwen2.5-3B, one T4, FP16, TP=1, V0/XFORMERS, eager, prefix caching and chunked prefill disabled.

| Capability | Status | Evidence / reason |
| --- | --- | --- |
| single_gpu_fp16_vllm | VALIDATED | Baseline startup log and successful real streaming generation; chunked prefill disabled. |
| streaming | VALIDATED | Baseline startup log and successful real streaming generation; chunked prefill disabled. |
| torch_compile | VALIDATED | Executed on T4 and passed numerical correctness checks. |
| triton | VALIDATED | Scoped to minimal add and small RMSNorm; the chunked-prefill attention kernel in attempt b remains FAILED. |
| tp2 | VALIDATED | Same FP16 checkpoint, two T4s, TP=2, mp executor, real streaming generation; no speedup claim. |
| awq | FAILED | RuntimeError: GPU compute processes remain or cannot be inspected: 832
 |

Full details, limitations, versions, and artifact references: `qualification.json`. The failed chunked-prefill attempt b remains preserved. No benchmark performance claim is made.
