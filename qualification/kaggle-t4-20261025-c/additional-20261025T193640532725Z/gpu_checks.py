
import sys
import torch

mode = sys.argv[1]
torch.manual_seed(42)
assert torch.cuda.get_device_capability(0) == (7, 5)

def reference(x, w):
    z = x.float()
    return (
        z * torch.rsqrt((z*z).mean(-1, keepdim=True) + 1e-6)
        * w.float()
    ).to(x.dtype)

if mode == "torch_compile":
    torch._dynamo.config.suppress_errors = False
    compiled = torch.compile(
        reference, backend="inductor", fullgraph=True
    )
    for seed in (7, 42):
        torch.manual_seed(seed)
        for shape in ((4, 128), (16, 2048)):
            x = torch.randn(
                shape, device="cuda", dtype=torch.float16
            )
            w = torch.randn(
                shape[-1], device="cuda", dtype=torch.float16
            )
            torch.testing.assert_close(
                reference(x, w),
                torch.nn.functional.rms_norm(
                    x.float(), (shape[-1],), w.float(), eps=1e-6
                ).half(),
                rtol=2e-3, atol=2e-3,
            )
            # Compilation and a subsequent execution must both work.
            for _ in range(2):
                y = compiled(x, w)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    y, reference(x, w), rtol=2e-3, atol=2e-3
                )
else:
    import triton
    import triton.language as tl

    @triton.jit
    def add(X, Y, Z, N: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        x = tl.load(X+i, i<N, 0)
        y = tl.load(Y+i, i<N, 0)
        tl.store(Z+i, x+y, i<N)

    @triton.jit
    def rms(X, W, Y, N: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        i = tl.arange(0, B)
        x = tl.load(X+row*N+i, i<N, 0).to(tl.float32)
        w = tl.load(W+i, i<N, 0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x*x, 0)/N + 1e-6) * w
        tl.store(Y+row*N+i, y, i<N)

    if mode == "triton_minimal":
        x = torch.randn(
            1025, device="cuda", dtype=torch.float16
        )
        y = torch.randn_like(x)
        z = torch.empty_like(x)
        add[(triton.cdiv(x.numel(), 128),)](
            x, y, z, x.numel(), 128,
            num_warps=4, num_stages=1,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(z, x+y, rtol=0, atol=0)

    elif mode == "triton_rmsnorm":
        for seed in (7, 42):
            torch.manual_seed(seed)
            for shape in ((4, 128), (16, 2048)):
                x = torch.randn(
                    shape, device="cuda", dtype=torch.float16
                )
                w = torch.randn(
                    shape[-1], device="cuda", dtype=torch.float16
                )
                y = torch.empty_like(x)
                rms[(shape[0],)](
                    x, w, y, shape[-1],
                    triton.next_power_of_2(shape[-1]),
                    num_warps=4, num_stages=1,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    y, reference(x, w), rtol=2e-3, atol=2e-3
                )
    else:
        raise ValueError(mode)

print(
    mode + ": VALIDATED; correctness only, no performance measurement",
    flush=True,
)
