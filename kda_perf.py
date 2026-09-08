#!/usr/bin/env python3
"""不用 ATK 的 KDA Ascend C 算子性能测试脚本。

通过 torch_npu Event 测量设备侧耗时，支持 chunk_kda_fwd / recurrent_kda /
kda_gate_cumsum 三个算子，供快速粗测与 A/B 对比使用。

注意：仓内官方性能口径是 msopprof 的 op_summary 中 Task Duration(us)
（见 docs/agents/03-方案设计.md），本脚本结果只作粗略体感；最终结论请用
msopprof 复测（文末 hint 给出命令）。

用法示例：
  python3 kda_perf.py --op chunk_kda_fwd --t 8192 --h 96 --hv 96
  python3 kda_perf.py --op chunk_kda_fwd --t 1024 --h 96 \
      --use-gate-in-kernel --safe-gate --varlen --check
  python3 kda_perf.py --op recurrent_kda --b 2 --t 2
  python3 kda_perf.py --op kda_gate_cumsum --t 8192 --hv 96 --use-gate-in-kernel

msopprof 复测（官方口径）：
  msopprof --aic-metrics=BasicInfo \
      --application="python3 kda_perf.py --op chunk_kda_fwd --t 8192 --h 96" \
      --output=./prof_out
  随后在输出目录的 op_summary*.csv 中查看各 kernel 的 Task Duration(us)。
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

OPS = ("chunk_kda_fwd", "recurrent_kda", "kda_gate_cumsum")

# 参考 ATK 模型 case（tests/atk/chunk_kda_fwd/atk_chunk_kda_fwd_perf.json）：
#   B=1, H=96, HV=96, T=1024, K=V=128, chunk=64, BSND, BF16, varlen,
#   safe_gate=true, use_gate_in_kernel=true, state_v_first=true
MODEL_CASE_HINT = (
    "复刻 ATK 模型 case 可用：--t 1024 --h 96 --hv 96 --varlen "
    "--use-gate-in-kernel --safe-gate --state-v-first"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="不用 ATK 测试 KDA Ascend C 算子性能（torch_npu Event 设备侧计时）",
        epilog=f"提示：{MODEL_CASE_HINT}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--op", choices=OPS, default="chunk_kda_fwd", help="被测算子")
    parser.add_argument("--b", type=int, default=1, help="batch（TND/NTD 必须为 1）")
    parser.add_argument(
        "--t", type=int, default=None,
        help="序列长度（默认：chunk_kda_fwd/kda_gate_cumsum 8192，recurrent_kda 2）",
    )
    parser.add_argument(
        "--h", type=int, default=None,
        help="query/key head 数（默认：chunk_kda_fwd 96，recurrent_kda 2；gate_cumsum 不使用）",
    )
    parser.add_argument("--hv", type=int, default=None, help="value/gate head 数（默认与 --h 相同）")
    parser.add_argument("--k", dest="kdim", type=int, default=128, help="K 维度")
    parser.add_argument("--v", dest="vdim", type=int, default=128, help="V 维度")
    parser.add_argument(
        "--dtype", choices=("bf16", "fp16"), default="bf16",
        help="q/k/v dtype（recurrent_kda 仅支持 bf16，会被强制）",
    )
    parser.add_argument(
        "--layout", choices=("BSND", "BNSD", "TND", "NTD"), default="BSND",
        help="输入 layout（recurrent_kda/kda_gate_cumsum 仅支持 BSND/TND）",
    )
    parser.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--scale", type=float, default=None, help="qk scale，默认 K**-0.5")
    parser.add_argument(
        "--varlen", action="store_true",
        help="chunk_kda_fwd/kda_gate_cumsum：cu_seqlens=[0,T] 单条变长序列（rank-4 要求 B=1）",
    )
    parser.add_argument(
        "--use-gate-in-kernel", action="store_true",
        help="g 按 raw gate 解释；自动生成 A_log [HV] 与 dt_bias [HV*K]",
    )
    parser.add_argument("--safe-gate", action="store_true")
    parser.add_argument("--lower-bound", type=float, default=-5.0, help="safe gate 下界，[-5,0)")
    parser.add_argument("--output-final-state", action="store_true")
    parser.add_argument("--disable-recompute", action="store_true", help="仅 chunk_kda_fwd")
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--use-beta-sigmoid", action="store_true", help="仅 recurrent_kda")
    parser.add_argument("--warmup", type=int, default=5, help="预热次数")
    parser.add_argument("--count", type=int, default=20, help="采样次数")
    parser.add_argument("--device", type=int, default=0, help="NPU 设备 id（未设置 ASCEND_RT_VISIBLE_DEVICES 时生效）")
    parser.add_argument("--check", action="store_true", help="额外执行一次并检查输出 finite")
    parser.add_argument("--verbose", action="store_true", help="逐次打印耗时")
    parser.add_argument("--csv", type=str, default=None, help="把逐次耗时写入 CSV 文件")
    args = parser.parse_args()

    # 按算子补默认 shape
    if args.t is None:
        args.t = 2 if args.op == "recurrent_kda" else 8192
    if args.h is None:
        args.h = 2 if args.op == "recurrent_kda" else 96
    if args.hv is None:
        args.hv = args.h
    if args.op == "kda_gate_cumsum":
        args.h = args.hv
    if args.scale is None:
        args.scale = args.kdim ** -0.5
    if args.warmup < 0 or args.count < 1:
        parser.error("--warmup 必须 >= 0，--count 必须 >= 1")
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.op == "recurrent_kda":
        if args.dtype != "bf16":
            print("[warn] recurrent_kda 仅支持 bf16，已强制 --dtype bf16")
            args.dtype = "bf16"
        if args.layout not in ("BSND", "TND"):
            sys.exit("[error] recurrent_kda 仅支持 BSND/TND layout")
        if (args.kdim, args.vdim) not in ((128, 128), (128, 256)):
            sys.exit("[error] recurrent_kda 仅支持 K=128, V=128 或 K=128, V=256")
        if args.t > 8:
            sys.exit("[error] recurrent_kda 每条序列长度必须 <= 8，当前 T=%d" % args.t)
        if args.varlen:
            sys.exit("[error] recurrent_kda 不支持 --varlen（本脚本按 dense 序列构造）")
        if args.disable_recompute:
            print("[warn] --disable-recompute 仅对 chunk_kda_fwd 生效，已忽略")
    elif args.op == "kda_gate_cumsum":
        if args.layout not in ("BSND", "TND"):
            sys.exit("[error] kda_gate_cumsum 仅支持 BSND/TND layout")
        if args.use_beta_sigmoid or args.output_final_state or args.state_v_first:
            sys.exit("[error] kda_gate_cumsum 不支持 --use-beta-sigmoid/--output-final-state/--state-v-first")

    if args.op != "recurrent_kda":
        if args.kdim < 16 or args.kdim > 256 or args.kdim % 16:
            sys.exit("[error] K 必须是 [16,256] 内 16 的倍数")
        if args.vdim < 16 or args.vdim > 256 or args.vdim % 16:
            sys.exit("[error] V 必须是 [16,256] 内 16 的倍数")
    if args.h <= 0 or args.hv < args.h or args.hv % args.h or args.hv > 128:
        sys.exit("[error] 需满足 0 < H <= HV <= 128 且 HV % H == 0")
    if args.layout in ("TND", "NTD") and args.b != 1:
        sys.exit("[error] %s layout 要求 B=1" % args.layout)
    if args.varlen and args.b != 1:
        sys.exit("[error] rank-4 变长输入要求 B=1")
    if args.use_gate_in_kernel and args.safe_gate and not (-5.0 <= args.lower_bound < 0.0):
        sys.exit("[error] safe gate 时 lower_bound 必须在 [-5, 0) 内")


def build_case(args: argparse.Namespace):
    """构造输入并返回 (call, outputs_check, config_lines)。"""
    b, t, h, hv = args.b, args.t, args.h, args.hv
    kdim, vdim = args.kdim, args.vdim
    layout = args.layout
    device = "npu"

    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as error:
        sys.exit("[error] 加载 torch/torch_npu 失败，请在 NPU 服务器上运行: %s" % error)
    try:
        from fla_npu.ops.ascendc import chunk_kda_fwd, kda_gate_cumsum, recurrent_kda
    except Exception as error:
        sys.exit("[error] 导入 fla_npu 失败，请先确认 wheel 已安装且 OPP 环境正常: %s" % error)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    is_rank3 = layout in ("TND", "NTD")
    is_sequence_major = layout in ("BSND", "TND")

    if is_sequence_major:
        q_shape = (t, h, kdim) if is_rank3 else (b, t, h, kdim)
        v_shape = (t, hv, vdim) if is_rank3 else (b, t, hv, vdim)
        g_shape = (t, hv, kdim) if is_rank3 else (b, t, hv, kdim)
        beta_shape = (t, hv) if is_rank3 else (b, t, hv)
    else:  # BNSD / NTD（head major）
        q_shape = (h, t, kdim) if is_rank3 else (b, h, t, kdim)
        v_shape = (hv, t, vdim) if is_rank3 else (b, hv, t, vdim)
        g_shape = (hv, t, kdim) if is_rank3 else (b, hv, t, kdim)
        beta_shape = (hv, t) if is_rank3 else (b, hv, t)

    q = torch.randn(q_shape, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(v_shape, device=device, dtype=dtype)

    a_log = None
    dt_bias = None
    if args.use_gate_in_kernel:
        # raw gate：与 ATK 模型 case 类似，带 bias 的 log gate 前体
        g = torch.randn(g_shape, device=device, dtype=torch.float32)
        a_log = torch.randn(hv, device=device, dtype=torch.float32) * 0.12
        dt_bias = -3.0 + 1.65 * torch.randn(hv * kdim, device=device, dtype=torch.float32)
    else:
        # 已激活的负自然对数 gate
        g = -0.01 * torch.rand(g_shape, device=device, dtype=torch.float32)
    beta = torch.randn(beta_shape, device=device, dtype=torch.float32)

    cu_seqlens = None
    if args.varlen and args.op != "recurrent_kda":
        cu_seqlens = [0, t]

    if args.op == "chunk_kda_fwd":
        def call():
            return chunk_kda_fwd(
                q, k, v, g, beta, args.scale, args.chunk_size,
                layout=layout,
                initial_state=None,
                output_final_state=args.output_final_state,
                cu_seqlens=cu_seqlens,
                safe_gate=args.safe_gate,
                lower_bound=args.lower_bound,
                use_gate_in_kernel=args.use_gate_in_kernel,
                A_log=a_log,
                dt_bias=dt_bias,
                disable_recompute=args.disable_recompute,
                return_intermediate_states=False,
                state_v_first=args.state_v_first,
            )

        def check_outputs(outputs):
            attn_out = outputs[0]
            return torch.isfinite(attn_out.float()).all().item()

    elif args.op == "recurrent_kda":
        cu = torch.arange(b + 1, device=device, dtype=torch.int32) * t
        state_shape = (
            (b, hv, vdim, kdim) if args.state_v_first else (b, hv, kdim, vdim)
        )
        state = torch.zeros(state_shape, device=device, dtype=torch.float32)
        safe_gate = args.safe_gate and args.use_gate_in_kernel

        def call():
            return recurrent_kda(
                q, k, v, g, beta, state,
                cu_seqlens=cu,
                A_log=a_log,
                dt_bias=dt_bias,
                layout=layout,
                scale=args.scale,
                output_final_state=args.output_final_state,
                use_gate_in_kernel=args.use_gate_in_kernel,
                use_beta_sigmoid_in_kernel=args.use_beta_sigmoid,
                safe_gate=safe_gate,
                lower_bound=args.lower_bound,
                state_v_first=args.state_v_first,
            )

        def check_outputs(outputs):
            return torch.isfinite(outputs[0].float()).all().item()

    else:  # kda_gate_cumsum
        def call():
            return kda_gate_cumsum(
                g, args.chunk_size,
                A_log=a_log,
                dt_bias=dt_bias,
                cu_seqlens=cu_seqlens,
                use_gate_in_kernel=args.use_gate_in_kernel,
                safe_gate=args.safe_gate,
                lower_bound=args.lower_bound,
            )

        def check_outputs(outputs):
            return torch.isfinite(outputs.float()).all().item()

    # 首次调用 + 同步：触发 executor/workspace 创建并确认可运行
    call()
    torch.npu.synchronize()

    return call, check_outputs


def run_bench(call, warmup: int, count: int, verbose: bool):
    import torch

    times_ms = []
    wall_start = time.perf_counter()
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    events = []
    for _ in range(count):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        events.append((start, end))
    torch.npu.synchronize()
    wall_end = time.perf_counter()
    for index, (start, end) in enumerate(events, 1):
        ms = start.elapsed_time(end)
        times_ms.append(ms)
        if verbose:
            print("  iter %3d: %.3f ms" % (index, ms))
    return times_ms, wall_end - wall_start


def main():
    args = parse_args()
    validate_args(args)

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible in (None, ""):
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    elif visible != str(args.device):
        print("[warn] ASCEND_RT_VISIBLE_DEVICES=%s 已设置，忽略 --device %d"
              % (visible, args.device))

    call, check_outputs = build_case(args)

    import torch
    try:
        device_name = torch.npu.get_device_name(0)
    except Exception:
        device_name = "npu:%d" % args.device

    print("[config] op=%s device=%s layout=%s dtype=%s chunk_size=%d scale=%.6f"
          % (args.op, device_name, args.layout, args.dtype, args.chunk_size, args.scale))
    print("[shape] B=%d T=%d H=%d HV=%d K=%d V=%d"
          % (args.b, args.t, args.h, args.hv, args.kdim, args.vdim))
    print("[flags] use_gate_in_kernel=%s safe_gate=%s varlen=%s output_final_state=%s "
          "disable_recompute=%s state_v_first=%s"
          % (args.use_gate_in_kernel, args.safe_gate, args.varlen,
             args.output_final_state, args.disable_recompute, args.state_v_first))
    print("[run] warmup=%d count=%d" % (args.warmup, args.count))

    if args.check:
        outputs = call()
        torch.npu.synchronize()
        try:
            ok = check_outputs(outputs)
            print("[check] 输出 finite: %s" % ("OK" if ok else "FAIL（存在 NaN/Inf）"))
        except Exception as error:
            print("[check] 输出检查失败: %s" % error)

    times_ms, wall_sec = run_bench(call, args.warmup, args.count, args.verbose)
    avg = statistics.mean(times_ms)
    p50 = statistics.median(times_ms)
    print("[result] avg=%.3f ms  min=%.3f ms  max=%.3f ms  p50=%.3f ms"
          % (avg, min(times_ms), max(times_ms), p50))
    print("[wall] 整个测量段 wall time %.3f s（含 Python 调度，不作为性能口径）" % wall_sec)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["iter", "ms"])
            for index, ms in enumerate(times_ms, 1):
                writer.writerow([index, "%.6f" % ms])
        print("[csv] 逐次耗时已写入 %s" % args.csv)

    print("[hint] 以上为 torch_npu Event 设备侧计时，仅作快速对比。")
    print("[hint] 仓内官方口径是 msopprof op_summary 的 Task Duration(us)，例如：")
    print('       msopprof --aic-metrics=BasicInfo \\')
    print('           --application="%s" \\' % " ".join(sys.argv))
    print('           --output=./prof_out')


if __name__ == "__main__":
    main()
