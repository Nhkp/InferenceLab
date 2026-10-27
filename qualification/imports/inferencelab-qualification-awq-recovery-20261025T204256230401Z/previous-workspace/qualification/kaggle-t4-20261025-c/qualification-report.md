# Qualification kaggle-t4-20261025-c

Baseline FP16 streaming and TP=2 streams were verified against their recorded events, monotone timestamps, finish reason, server usage, and recalculated client metrics. Compile and Triton correctness results are supported by the preserved GPU check source and logs.

| Capability | Status | Evidence / reason |
| --- | --- | --- |
| `single_gpu_fp16_vllm` | VALIDATED | Baseline startup log and successful real streaming generation; chunked prefill disabled. |
| `streaming` | VALIDATED | Baseline startup log and successful real streaming generation; chunked prefill disabled. |
| `torch_compile` | VALIDATED | Executed on T4 and passed numerical correctness checks. |
| `triton` | VALIDATED | Scoped to minimal add and small RMSNorm; the chunked-prefill attention kernel in attempt b remains FAILED. |
| `tp2` | VALIDATED | Same FP16 checkpoint, two T4s, TP=2, mp executor, real streaming generation; no speedup claim. |
| `awq` | NOT_TESTED | GPU-idle prerequisite failed because TP=2 engine PID 832 remained. AWQ metadata, loading, quantization, and generation were not attempted. |

## Qualified baseline

Kaggle Tesla T4; Python 3.12.12; torch 2.8.0 / CUDA runtime 12.8; vLLM 0.10.2 V0; Triton 3.4.0; Transformers 4.55.2; XFORMERS 0.0.32.post1. Model/tokenizer: Qwen/Qwen2.5-3B-Instruct at aa8e72537993ba99e69dfaafa59ed015b17504d1. FP16, TP=1, eager, prefix caching disabled, **chunked prefill disabled**. The TP=2 check uses the same checkpoint/dtype on two T4s with the mp executor.

## Correction and remaining work

Qualification remains open for AWQ. Its original FAILED label represented a blocked prerequisite, not attempted AWQ inference. The original manifest/report are retained as source-archive-qualification.json and source-archive-report.md; the unchanged source ZIP and its SHA256 are preserved under ../imports/. The TP=2 startup log identifies engine PID 832, which was still using a GPU before AWQ could start. Cleanup must verify ownership and stop remaining workers before AWQ qualification; no passed GPU checks need repeating.

Attempt b's chunked-prefill attention kernel remains FAILED (98304 bytes requested versus 65536 available shared memory). Passing the minimal Triton and RMSNorm checks does not validate that kernel. Measurements are single-request functional smokes, with no performance, SLO, p99, or TP speedup claim. Runtime KV dtype is not independently introspected; upstream cancellation/error paths remain untested. No platform implementation has started.
