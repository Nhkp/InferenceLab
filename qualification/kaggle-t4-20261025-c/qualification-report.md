# Qualification kaggle-t4-20261025-c

**00B complete.** The final Kaggle artifacts have been imported and reviewed. No further platform stage has been implemented.

| Capability | Status |
| --- | --- |
| `single_gpu_fp16_vllm` | VALIDATED |
| `streaming` | VALIDATED |
| `torch_compile` | VALIDATED |
| `triton` | VALIDATED |
| `tp2` | VALIDATED |
| `awq` | VALIDATED |

## Qualified configuration

Kaggle Tesla T4, SM 7.5; baseline GPU 0, FP16, TP=1; vLLM **0.10.2 V0**, **XFORMERS**, eager mode; prefix caching **disabled**, chunked prefill **disabled**. Python **3.12.12**, torch **2.8.0** / CUDA runtime **12.8**, Triton **3.4.0**, Transformers **4.55.2**, xformers **0.0.32.post1**. Full installed package inventory is in pip-freeze.txt and the additional batch inventory; no project requirements were changed by this review.

Model/tokenizer: `Qwen/Qwen2.5-3B-Instruct` at `aa8e72537993ba99e69dfaafa59ed015b17504d1`. TP=2 used the same checkpoint and dtype on two T4s with the mp executor.

AWQ model/tokenizer: `Qwen/Qwen2.5-3B-Instruct-AWQ` at `3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd`; 4-bit AWQ with FP16 activations, one T4 and TP=1. Publisher source metadata, architecture, vocabulary and chat template checks passed. Effective `quantization=awq` is recorded in startup logs. The publisher does not pin the original quantization-source commit; no exact-commit causal performance comparison is certified.

## Evidence and verification

All nine required artifacts exist for attempts b and c. Baseline, TP=2, and both AWQ recovery streams were checked for valid termination, expected finish reason, 128 server-reported output tokens, recorded content-event timestamps, and recalculation of client_ttft_ms, client_e2e_ms and client_tpot_estimate_ms. Compile/Triton source and logs preserve the numerical checks for two seeds and two small RMSNorm shapes. Paths to each capability's evidence and the final archive SHA256 are in qualification.json. The unchanged source ZIP is preserved under ../imports/ with an import audit; conflicting earlier local files were backed up.

## Failures and limits preserved

Attempt b's chunked-prefill Triton attention kernel failed: 98304 bytes shared memory requested versus a 65536-byte launch limit. Minimal Triton/RMSNorm success does not validate that kernel. All serving checks in c explicitly disable chunked prefill.

The initial AWQ FAILED label represented a GPU-idle prerequisite blocked by leftover TP=2 engine PID 832. Recovery verified ownership and corrected cleanup before actual AWQ execution. Earlier manifests, reports, and the original archive remain preserved.

These are functional and correctness smokes, not performance/SLO/p99 certification or proof of TP/AWQ speedup. Exact resolved KV dtype and upstream cancellation/error paths remain untested. The Kaggle environment has no repository Git checkout; its Git SHA remains null with a reason and the qualification script hash is preserved. Repository metadata still specifies Python >=3.14; the later bootstrap must explicitly reconcile it with qualified Python 3.12 rather than infer GPU support for 3.14.
