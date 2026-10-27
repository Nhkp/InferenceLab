# Qualification kaggle-t4-20261025-c

| Capability | Status | Reason |
| --- | --- | --- |
| single_gpu_fp16_vllm | VALIDATED | Baseline startup log and successful real streaming generation; chunked prefill disabled. |
| streaming | VALIDATED | Baseline startup log and successful real streaming generation; chunked prefill disabled. |
| torch_compile | VALIDATED | Executed on T4 and passed numerical correctness checks. |
| triton | VALIDATED | Scoped to minimal add and small RMSNorm; the chunked-prefill attention kernel in attempt b remains FAILED. |
| tp2 | VALIDATED | Same FP16 checkpoint, two T4s, TP=2, mp executor, real streaming generation; no speedup claim. |
| awq | VALIDATED | Publisher-linked same-source AWQ, matching tokenizer/template, effective quantization=awq, FP16 activations, TP=1, real stream; no performance claim. |

The earlier AWQ result was a blocked prerequisite, not AWQ execution. Its report and archive remain preserved. Baseline serving requires chunked prefill disabled; the earlier enabled configuration failed. Only correctness and functional smoke checks were performed; no performance claim is made. Full limitations and evidence references are in qualification.json.
