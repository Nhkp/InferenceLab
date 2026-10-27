# Qualification kaggle-t4-20261025-a

Qualification is **open**, not completed. This ID preserves a local attempt; it does not certify Kaggle/T4 execution.

## Scope and evidence

Read START_HERE, PDD v1.2, benchmark contract 1.0, global rules, and prompt 00B. Only 00B was executed. No platform implementation, package installation, model download, or configuration change was performed.

Observed local Linux host, Python 3.14.4 in `.venv`, and Git `1b57fb29b54d3c74a441a8b0fa06dfd10b22877a`. `nvidia-smi` and topology commands could not execute; their errors are preserved. No NVIDIA device nodes were visible. NVIDIA PCI observations are recorded in `environment.json`; lack of access does not certify physical GPU counts. Package versions are observed metadata, never validated compatibility. `pip-freeze.txt` records the interpreter and actual inventory command, including any fallback/error.

## Capability statuses

| Capability | Status | Reason |
| --- | --- | --- |
| Reference Kaggle/T4 access | UNAVAILABLE | No accessible reference GPU session. |
| `single_gpu_fp16_vllm` | NOT_TESTED | Reference Kaggle/T4 inaccessible. |
| `streaming` | NOT_TESTED | Reference Kaggle/T4 inaccessible. |
| `torch_compile` | NOT_TESTED | Reference Kaggle/T4 inaccessible. |
| `triton` | NOT_TESTED | Reference Kaggle/T4 inaccessible. |
| `tp2` | NOT_TESTED | Reference Kaggle/T4 inaccessible. |
| `awq` | NOT_TESTED | Reference Kaggle/T4 inaccessible. |
| `client_ttft_ms` | NOT_TESTED | No stream or monotone timestamps. |
| `client_e2e_ms` | NOT_TESTED | No valid stream termination. |
| `client_tpot_estimate_ms` | NOT_TESTED | No content events or server output-token usage. |

UNAVAILABLE describes the missing execution resource; NOT_TESTED describes capabilities whose GPU checks did not run. No capability is VALIDATED or FAILED because no capability execution established success or incompatibility.

## Assumptions and unresolved decisions

Serving reference remains Kaggle/T4, baseline one GPU, FP16, TP=1. TP=2 requires two observed T4s and the same checkpoint/dtype. The initial model candidate remains `Qwen/Qwen2.5-3B-Instruct`; immutable model/tokenizer revisions are unresolved. No AWQ variant or attention backend was selected. Installed Python 3.14.4 differs from the PDD first candidate 3.12; neither metadata nor interpreter was changed. `validated_versions` is empty.

Triton was not tested: no T4 incompatibility conclusion or alternate-GPU contingency is justified by this attempt. Missing metrics remain null with reasons; no synthetic stream substitutes for GPU evidence.

## Artifacts and continuation

All nine files required by 00B are preserved. The server command, server log, and stream sample files explicitly record non-execution rather than simulated output. `command-results.json` additionally preserves commands, exit codes, stdout, and stderr.

To complete 00B, execute in an accessible Kaggle/T4 session with an isolated Python 3.12 candidate environment, explicitly resolve immutable revisions and candidate versions, preserve effective runtime settings and startup logs, then run baseline streaming/usage/EOS checks, correctness plus post-compilation execution for torch.compile, a minimal Triton kernel and RMSNorm if possible, TP=2 streaming if two T4s are available, and same-source AWQ loading/streaming with observable quantization. Preserve failures without fallback. Capture timestamps with a monotone clock; use server usage tokens, never SSE event counts, for the contract's TPOT estimate. Keep this unavailable attempt and use a new qualification ID for the GPU execution.

The 00B definition of done is unmet. No later implementation stage has started.
