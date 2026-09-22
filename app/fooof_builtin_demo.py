#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
内置合成功率谱演示。

用途：
1. 不需要EEG文件即可确认环境是否正常；
2. 直观看到“非周期背景 + 周期峰 = 完整频谱”；
3. 熟悉FOOOF的主要输出参数。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fooof import FOOOF


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def open_folder(path: Path) -> None:
    if os.name == "nt":
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except OSError:
            pass


def main() -> int:
    rng = np.random.default_rng(20260806)
    freqs = np.arange(1.0, 40.01, 0.25)

    # 在log10功率空间构造非周期背景
    aperiodic_log = 1.1 - 1.5 * np.log10(freqs)

    # 构造两个周期峰：约10 Hz与20 Hz
    alpha_peak = 0.40 * np.exp(-0.5 * ((freqs - 10.0) / 1.2) ** 2)
    beta_peak = 0.20 * np.exp(-0.5 * ((freqs - 20.0) / 2.0) ** 2)

    noise = rng.normal(0.0, 0.012, size=freqs.size)
    spectrum = 10 ** (aperiodic_log + alpha_peak + beta_peak + noise)

    model = FOOOF(
        peak_width_limits=[1.0, 8.0],
        max_n_peaks=6,
        min_peak_height=0.05,
        peak_threshold=2.0,
        aperiodic_mode="fixed",
        verbose=False,
    )
    model.fit(freqs, spectrum, [2, 40])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PACKAGE_ROOT / "results" / f"builtin_demo_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 输入功率谱
    np.savetxt(
        out_dir / "01_input_spectrum.csv",
        np.column_stack([freqs, spectrum]),
        delimiter=",",
        header="frequency_hz,power_linear",
        comments="",
        fmt="%.12e",
    )

    # 非周期参数
    ap = np.asarray(model.aperiodic_params_, dtype=float)
    with (out_dir / "02_aperiodic_parameters.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["offset", "exponent", "r_squared", "error"])
        writer.writerow([ap[0], ap[1], model.r_squared_, model.error_])

    # 周期峰参数
    with (out_dir / "03_periodic_peak_parameters.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["peak_index", "center_frequency_hz",
             "peak_power_above_background_log10", "bandwidth_hz"]
        )
        for index, peak in enumerate(model.peak_params_, start=1):
            writer.writerow([index, peak[0], peak[1], peak[2]])

    # 拟合图
    model.plot(plot_peaks="shade", plot_aperiodic=True, add_legend=True)
    plt.title("Built-in synthetic spectrum: FOOOF fit")
    plt.tight_layout()
    plt.savefig(out_dir / "04_model_fit.png", dpi=220, bbox_inches="tight")
    plt.close("all")

    explanation = f"""内置示例运行成功

输出目录：
{out_dir}

非周期参数：
offset = {ap[0]:.6f}
exponent = {ap[1]:.6f}

检测到的周期峰数量：
{model.n_peaks_}

拟合质量：
R² = {model.r_squared_:.6f}
error = {model.error_:.6f}

这是一条人为构造的合成功率谱，仅用于教学和环境测试。
"""
    (out_dir / "README.txt").write_text(explanation, encoding="utf-8")

    print("=" * 60)
    print("内置演示运行成功")
    print("=" * 60)
    print(explanation)
    print("建议先打开：04_model_fit.png")

    open_folder(out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("内置演示运行失败：", exc)
        raise
