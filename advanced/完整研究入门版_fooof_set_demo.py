#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEGLAB .set → Welch功率谱 → FOOOF 1.1.1频谱参数化（教学版）

输入
----
一个已经预处理好的 EEGLAB .set 文件。
连续数据和分段（epochs）数据均会尝试读取。

输出
----
1. FOOOF所需的频率和线性功率谱数据
2. 平均功率谱的周期/非周期拟合结果
3. 每个通道的非周期参数与周期峰参数
4. 平均功率谱拟合图
5. 可重新载入的FOOOF模型文件和说明文件

安装依赖
--------
python -m pip install mne scipy matplotlib numpy fooof==1.1.1

运行方法
--------
方法一：在命令行指定文件
python fooof_set_demo.py "D:\\data\\subject01.set"

方法二：直接运行脚本，然后在弹出的窗口中选择 .set 文件

注意
----
- FOOOF接收的是“线性尺度的功率谱”，不是原始EEG，也不是dB或log功率。
- 本脚本先用Welch方法计算PSD，再将PSD传入FOOOF。
- 默认拟合2–40 Hz，并使用fixed非周期模型；这些设置仅用于教学演示，
  正式研究前需要根据数据、频率分辨率和拟合质量重新确定。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

# 使用非交互式后端，避免服务器或无图形环境下报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

try:
    import mne
except ImportError as exc:
    raise SystemExit(
        "缺少MNE。请先运行：python -m pip install mne"
    ) from exc

try:
    from fooof import FOOOF
except ImportError as exc:
    raise SystemExit(
        "缺少FOOOF 1.1.1。请先运行：python -m pip install fooof==1.1.1"
    ) from exc


# ============================================================
# 一、教学演示参数：正式研究前应结合数据重新确定
# ============================================================

# 计算PSD时保留的频率范围
PSD_FMIN = 1.0
PSD_FMAX = 40.0

# FOOOF实际拟合范围
FIT_FMIN = 2.0
FIT_FMAX = 40.0

# Welch分段长度。连续数据优先使用4秒；若epoch更短，则使用完整epoch长度
WELCH_SEGMENT_SECONDS = 4.0
WELCH_OVERLAP_RATIO = 0.5

# FOOOF设置
PEAK_WIDTH_LIMITS = (1.0, 8.0)
MAX_N_PEAKS = 6
MIN_PEAK_HEIGHT = 0.1
PEAK_THRESHOLD = 2.0
APERIODIC_MODE = "fixed"

# 是否同时拟合每个通道。True会输出通道级参数，但运行时间稍长
FIT_EACH_CHANNEL = True


def safe_column_name(name: str) -> str:
    """把通道名转换为适合CSV表头的形式。"""
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", str(name))
    return cleaned.strip("_") or "channel"


def make_unique_names(names: list[str]) -> list[str]:
    """避免重复通道名导致CSV列无法区分。"""
    counts: dict[str, int] = {}
    output: list[str] = []

    for name in names:
        base = safe_column_name(name)
        counts[base] = counts.get(base, 0) + 1
        suffix = "" if counts[base] == 1 else f"_{counts[base]}"
        output.append(base + suffix)

    return output


def choose_set_file() -> Path:
    """没有命令行参数时，弹出文件选择窗口。"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="选择预处理好的EEGLAB .set文件",
            filetypes=[("EEGLAB SET", "*.set"), ("所有文件", "*.*")],
        )
        root.destroy()
    except Exception as exc:
        raise RuntimeError(
            "无法打开文件选择窗口。请改用命令行运行：\n"
            'python fooof_set_demo.py "D:\\data\\subject01.set"'
        ) from exc

    if not selected:
        raise RuntimeError("没有选择任何文件。")

    return Path(selected)


def get_eeg_picks(info: mne.Info) -> np.ndarray:
    """
    优先选择MNE标记为EEG的通道。
    若文件中没有正确保存通道类型，则回退到“排除明显辅助通道”的策略。
    """
    picks = mne.pick_types(
        info,
        meg=False,
        eeg=True,
        eog=False,
        ecg=False,
        emg=False,
        stim=False,
        misc=False,
        exclude="bads",
    )

    if len(picks) > 0:
        return np.asarray(picks, dtype=int)

    warnings.warn(
        "文件中未找到明确标记为EEG的通道，将排除常见辅助通道后使用其余通道。"
    )

    excluded_tokens = (
        "EOG",
        "VEOG",
        "HEOG",
        "ECG",
        "EKG",
        "EMG",
        "TRIG",
        "TRIGGER",
        "STATUS",
        "EVENT",
        "STIM",
        "MARK",
    )
    bad_names = set(info.get("bads", []))

    fallback = [
        index
        for index, name in enumerate(info["ch_names"])
        if name not in bad_names
        and not any(token in name.upper() for token in excluded_tokens)
    ]

    if not fallback:
        raise RuntimeError("没有找到可以用于分析的EEG通道。")

    return np.asarray(fallback, dtype=int)


def load_eeglab_set(
    set_path: Path,
) -> tuple[np.ndarray, float, list[str], str, dict[str, Any]]:
    """
    尝试先按连续数据读取；失败后再按epochs数据读取。

    返回
    ----
    data:
        连续数据形状为 [通道, 时间点]；
        epochs数据形状为 [epoch, 通道, 时间点]。
    sfreq:
        采样率。
    channel_names:
        实际纳入分析的通道名。
    data_kind:
        continuous 或 epoched。
    extra_info:
        额外数据说明。
    """
    raw_error: Exception | None = None

    try:
        raw = mne.io.read_raw_eeglab(
            str(set_path),
            preload=True,
            verbose="ERROR",
        )
        picks = get_eeg_picks(raw.info)

        # 对标记为BAD的时间段进行省略；若当前MNE版本不支持，则读取全部数据
        try:
            data = raw.get_data(
                picks=picks,
                reject_by_annotation="omit",
            )
        except TypeError:
            data = raw.get_data(picks=picks)

        if data.size == 0:
            raise RuntimeError("读取后没有可用数据点。")

        channel_names = [raw.ch_names[index] for index in picks]
        extra_info = {
            "duration_seconds": float(data.shape[-1] / raw.info["sfreq"]),
            "n_epochs": None,
            "samples_per_epoch": None,
        }
        return (
            np.asarray(data, dtype=np.float64),
            float(raw.info["sfreq"]),
            channel_names,
            "continuous",
            extra_info,
        )

    except Exception as exc:
        raw_error = exc

    try:
        epochs = mne.read_epochs_eeglab(
            str(set_path),
            verbose="ERROR",
        )
        picks = get_eeg_picks(epochs.info)
        data = np.asarray(epochs.get_data(picks=picks), dtype=np.float64)

        if data.size == 0:
            raise RuntimeError("读取后没有可用epoch。")

        channel_names = [epochs.ch_names[index] for index in picks]
        extra_info = {
            "duration_seconds": float(
                data.shape[0] * data.shape[-1] / epochs.info["sfreq"]
            ),
            "n_epochs": int(data.shape[0]),
            "samples_per_epoch": int(data.shape[-1]),
        }
        return (
            data,
            float(epochs.info["sfreq"]),
            channel_names,
            "epoched",
            extra_info,
        )

    except Exception as epochs_error:
        raise RuntimeError(
            "无法读取该.set文件。\n\n"
            f"按连续数据读取时的错误：{raw_error}\n\n"
            f"按epochs数据读取时的错误：{epochs_error}\n\n"
            "请检查：\n"
            "1. .set对应的.fdt文件是否位于同一文件夹；\n"
            "2. 文件是否损坏；\n"
            "3. 是否使用了MNE暂不支持的EEGLAB保存形式。"
        ) from epochs_error


def compute_welch_psd(
    data: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    计算Welch PSD。

    返回
    ----
    freqs:
        频率数组，[频率点]。
    channel_psd:
        每个通道的PSD，[通道, 频率点]。
        若输入为epochs，则先对每个epoch计算，再在epoch维度求平均。
    mean_psd:
        所有通道在线性功率尺度下的平均PSD，[频率点]。
    welch_info:
        实际使用的Welch参数。
    """
    if data.ndim not in (2, 3):
        raise ValueError(
            f"数据维度应为2维或3维，当前形状为：{data.shape}"
        )

    n_times = int(data.shape[-1])
    requested_nperseg = max(8, int(round(WELCH_SEGMENT_SECONDS * sfreq)))
    nperseg = min(requested_nperseg, n_times)

    if nperseg < 8:
        raise RuntimeError("数据时间点过少，无法可靠计算功率谱。")

    noverlap = min(
        int(round(nperseg * WELCH_OVERLAP_RATIO)),
        nperseg - 1,
    )

    freqs, psd = welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
        average="mean",
    )

    # epochs输入：[epoch, channel, frequency] → 对epoch求平均
    if data.ndim == 3:
        channel_psd = np.nanmean(psd, axis=0)
    else:
        channel_psd = psd

    max_allowed_frequency = min(PSD_FMAX, sfreq / 2.0)
    mask = (freqs >= PSD_FMIN) & (freqs <= max_allowed_frequency)

    freqs = np.asarray(freqs[mask], dtype=np.float64)
    channel_psd = np.asarray(channel_psd[:, mask], dtype=np.float64)

    if freqs.size < 3:
        raise RuntimeError(
            "PSD频率点过少。请检查采样率、数据长度和频率范围设置。"
        )

    if not np.all(np.isfinite(channel_psd)):
        raise RuntimeError("PSD中出现NaN或无穷值，请检查输入数据。")

    # FOOOF要求功率为正数；仅对可能出现的数值零值做最小保护
    tiny = np.finfo(np.float64).tiny
    channel_psd = np.maximum(channel_psd, tiny)

    # 在“线性功率尺度”下对通道求平均
    mean_psd = np.mean(channel_psd, axis=0)

    welch_info = {
        "nperseg_samples": int(nperseg),
        "segment_seconds": float(nperseg / sfreq),
        "noverlap_samples": int(noverlap),
        "frequency_resolution_hz": float(freqs[1] - freqs[0]),
    }

    return freqs, channel_psd, mean_psd, welch_info


def new_fooof_model() -> FOOOF:
    """使用脚本顶部设置创建FOOOF模型。"""
    return FOOOF(
        peak_width_limits=PEAK_WIDTH_LIMITS,
        max_n_peaks=MAX_N_PEAKS,
        min_peak_height=MIN_PEAK_HEIGHT,
        peak_threshold=PEAK_THRESHOLD,
        aperiodic_mode=APERIODIC_MODE,
        verbose=False,
    )


def fit_one_spectrum(
    freqs: np.ndarray,
    spectrum: np.ndarray,
) -> tuple[FOOOF, tuple[float, float]]:
    """拟合一条线性功率谱。"""
    effective_fmax = min(FIT_FMAX, float(freqs[-1]))
    effective_range = (FIT_FMIN, effective_fmax)

    if effective_range[1] <= effective_range[0]:
        raise RuntimeError(
            f"可用最高频率仅为{effective_fmax:.3f} Hz，无法拟合"
            f"{FIT_FMIN:.1f} Hz以上范围。"
        )

    model = new_fooof_model()
    model.fit(
        np.asarray(freqs, dtype=np.float64),
        np.asarray(spectrum, dtype=np.float64),
        list(effective_range),
    )

    if not model.has_model:
        raise RuntimeError("FOOOF未能得到有效模型。")

    return model, effective_range


def aperiodic_result_row(
    model: FOOOF,
    scope: str,
    channel: str,
    status: str = "success",
    message: str = "",
) -> dict[str, Any]:
    """整理一条非周期参数和拟合质量记录。"""
    row: dict[str, Any] = {
        "scope": scope,
        "channel": channel,
        "offset": np.nan,
        "knee": np.nan,
        "exponent": np.nan,
        "r_squared": np.nan,
        "error": np.nan,
        "n_peaks": 0,
        "status": status,
        "message": message,
    }

    if status != "success":
        return row

    params = np.asarray(model.aperiodic_params_, dtype=float)

    if APERIODIC_MODE == "fixed":
        row["offset"] = float(params[0])
        row["exponent"] = float(params[1])
    else:
        row["offset"] = float(params[0])
        row["knee"] = float(params[1])
        row["exponent"] = float(params[2])

    row["r_squared"] = float(model.r_squared_)
    row["error"] = float(model.error_)
    row["n_peaks"] = int(model.n_peaks_)
    return row


def periodic_result_rows(
    model: FOOOF,
    scope: str,
    channel: str,
) -> list[dict[str, Any]]:
    """整理一条频谱中的全部周期峰。"""
    rows: list[dict[str, Any]] = []
    peaks = np.asarray(model.peak_params_, dtype=float)

    for peak_index, peak in enumerate(peaks, start=1):
        rows.append(
            {
                "scope": scope,
                "channel": channel,
                "peak_index": peak_index,
                "center_frequency_hz": float(peak[0]),
                "peak_power_above_aperiodic_log10": float(peak[1]),
                "bandwidth_hz": float(peak[2]),
            }
        )

    return rows


def write_dict_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """将字典列表保存为UTF-8 BOM CSV，便于Excel直接打开中文。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_outputs(
    set_path: Path,
    output_dir: Path,
    freqs: np.ndarray,
    channel_psd: np.ndarray,
    mean_psd: np.ndarray,
    channel_names: list[str],
    mean_model: FOOOF,
    effective_fit_range: tuple[float, float],
    sfreq: float,
    data_kind: str,
    extra_info: dict[str, Any],
    welch_info: dict[str, Any],
) -> None:
    """保存输入PSD、模型参数、拟合曲线、图像和说明文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_channel_names = make_unique_names(channel_names)

    # --------------------------------------------------------
    # 1. 保存FOOOF直接需要的线性PSD
    # --------------------------------------------------------
    psd_matrix = np.column_stack(
        [freqs, mean_psd, channel_psd.T]
    )
    psd_header = ",".join(
        ["frequency_hz", "mean_psd_linear"]
        + [f"{name}_psd_linear" for name in unique_channel_names]
    )
    np.savetxt(
        output_dir / "01_fooof_input_psd.csv",
        psd_matrix,
        delimiter=",",
        header=psd_header,
        comments="",
        fmt="%.12e",
    )

    np.savez_compressed(
        output_dir / "01_fooof_input_arrays.npz",
        freqs=freqs,
        mean_psd=mean_psd,
        channel_psd=channel_psd,
        channel_names=np.asarray(channel_names, dtype="U"),
    )

    # --------------------------------------------------------
    # 2. 保存平均频谱的拟合曲线
    #    FOOOF的模型在log10功率空间中进行加法分解
    # --------------------------------------------------------
    fitted_freqs = np.asarray(mean_model.freqs, dtype=float)
    curve_matrix = np.column_stack(
        [
            fitted_freqs,
            mean_model.get_data("full", "log"),
            mean_model.get_model("aperiodic", "log"),
            mean_model.get_model("peak", "log"),
            mean_model.get_model("full", "log"),
        ]
    )
    np.savetxt(
        output_dir / "02_mean_spectrum_model_curves.csv",
        curve_matrix,
        delimiter=",",
        header=(
            "frequency_hz,"
            "original_log10_psd,"
            "aperiodic_fit_log10,"
            "periodic_component_log10_above_background,"
            "full_model_fit_log10"
        ),
        comments="",
        fmt="%.12e",
    )

    # --------------------------------------------------------
    # 3. 平均频谱和每通道参数
    # --------------------------------------------------------
    aperiodic_rows = [
        aperiodic_result_row(
            mean_model,
            scope="channel_average",
            channel="MEAN",
        )
    ]
    periodic_rows = periodic_result_rows(
        mean_model,
        scope="channel_average",
        channel="MEAN",
    )

    if FIT_EACH_CHANNEL:
        print("正在拟合每个通道，请稍候……")
        for index, (channel, spectrum) in enumerate(
            zip(channel_names, channel_psd),
            start=1,
        ):
            print(
                f"\r  通道 {index}/{len(channel_names)}：{channel}",
                end="",
                flush=True,
            )
            try:
                channel_model, _ = fit_one_spectrum(freqs, spectrum)
                aperiodic_rows.append(
                    aperiodic_result_row(
                        channel_model,
                        scope="channel",
                        channel=channel,
                    )
                )
                periodic_rows.extend(
                    periodic_result_rows(
                        channel_model,
                        scope="channel",
                        channel=channel,
                    )
                )
            except Exception as exc:
                aperiodic_rows.append(
                    aperiodic_result_row(
                        mean_model,
                        scope="channel",
                        channel=channel,
                        status="failed",
                        message=str(exc),
                    )
                )
        print()

    write_dict_rows(
        output_dir / "03_aperiodic_parameters.csv",
        aperiodic_rows,
        [
            "scope",
            "channel",
            "offset",
            "knee",
            "exponent",
            "r_squared",
            "error",
            "n_peaks",
            "status",
            "message",
        ],
    )

    write_dict_rows(
        output_dir / "04_periodic_peak_parameters.csv",
        periodic_rows,
        [
            "scope",
            "channel",
            "peak_index",
            "center_frequency_hz",
            "peak_power_above_aperiodic_log10",
            "bandwidth_hz",
        ],
    )

    # --------------------------------------------------------
    # 4. 保存平均频谱拟合图
    # --------------------------------------------------------
    mean_model.plot(
        plot_peaks="shade",
        plot_aperiodic=True,
        add_legend=True,
    )
    plt.title(f"FOOOF fit: {set_path.stem} (channel average)")
    plt.tight_layout()
    plt.savefig(
        output_dir / "05_mean_spectrum_fooof_fit.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close("all")

    # 保存FOOOF自身的JSON格式，之后可以重新载入
    mean_model.save(
        "06_mean_fooof_model",
        file_path=str(output_dir),
        save_results=True,
        save_settings=True,
        save_data=True,
    )

    # --------------------------------------------------------
    # 5. 保存运行信息
    # --------------------------------------------------------
    metadata = {
        "input_file": str(set_path.resolve()),
        "data_kind": data_kind,
        "sampling_rate_hz": sfreq,
        "n_channels": len(channel_names),
        "channel_names": channel_names,
        "extra_data_info": extra_info,
        "psd_range_hz": [PSD_FMIN, min(PSD_FMAX, sfreq / 2.0)],
        "fooof_fit_range_hz": list(effective_fit_range),
        "welch": welch_info,
        "fooof_settings": {
            "package": "fooof",
            "expected_version": "1.1.1",
            "peak_width_limits_hz": list(PEAK_WIDTH_LIMITS),
            "max_n_peaks": MAX_N_PEAKS,
            "min_peak_height_log10": MIN_PEAK_HEIGHT,
            "peak_threshold_sd": PEAK_THRESHOLD,
            "aperiodic_mode": APERIODIC_MODE,
        },
        "notes": [
            "PSD以线性功率尺度输入FOOOF。",
            "offset受EEG单位、参考方式及总体功率尺度影响。",
            "当前设置用于教学演示，正式研究应统一参数并检查拟合质量。",
        ],
    }

    with (output_dir / "07_run_information.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    readme = f"""FOOOF教学脚本输出说明
========================

输入文件
--------
{set_path}

分析流程
--------
预处理后的.set
→ Welch功率谱
→ 线性PSD
→ FOOOF 1.1.1
→ 周期峰与非周期参数

主要文件
--------
01_fooof_input_psd.csv
    FOOOF需要的输入数据。
    frequency_hz为频率；其余列是线性尺度PSD。

01_fooof_input_arrays.npz
    可供Python直接载入：
        data = np.load("01_fooof_input_arrays.npz")
        freqs = data["freqs"]
        spectrum = data["mean_psd"]

02_mean_spectrum_model_curves.csv
    平均频谱的原始曲线、非周期拟合、周期成分和完整模型。
    这些列使用log10功率表示。

03_aperiodic_parameters.csv
    非周期参数：
        offset   = 非周期背景的整体高度
        exponent = 非周期背景下降的陡峭程度
        knee     = 仅knee模式下存在
        r_squared、error = 模型拟合质量指标

04_periodic_peak_parameters.csv
    每个周期峰的参数：
        center_frequency_hz = 中心频率CF
        peak_power_above_aperiodic_log10 = 高出非周期背景的峰值功率PW
        bandwidth_hz = 峰带宽BW

05_mean_spectrum_fooof_fit.png
    所有EEG通道平均PSD的FOOOF拟合图。

06_mean_fooof_model.json
    FOOOF模型、设置、输入数据与结果，可供以后重新载入。

07_run_information.json
    本次运行的数据和参数记录。

本次主要设置
------------
数据类型：{data_kind}
采样率：{sfreq:.3f} Hz
通道数：{len(channel_names)}
PSD范围：{PSD_FMIN:.1f}–{min(PSD_FMAX, sfreq / 2.0):.1f} Hz
FOOOF拟合范围：{effective_fit_range[0]:.1f}–{effective_fit_range[1]:.1f} Hz
Welch分段长度：{welch_info["segment_seconds"]:.3f} 秒
频率分辨率：{welch_info["frequency_resolution_hz"]:.6f} Hz
非周期模式：{APERIODIC_MODE}

重新使用保存的PSD
-----------------
from pathlib import Path
import numpy as np
from fooof import FOOOF

data = np.load(Path("01_fooof_input_arrays.npz"))
freqs = data["freqs"]
spectrum = data["mean_psd"]

fm = FOOOF(
    peak_width_limits=[1.0, 8.0],
    max_n_peaks=6,
    min_peak_height=0.1,
    peak_threshold=2.0,
    aperiodic_mode="fixed",
)

fm.report(freqs, spectrum, [2, 40])

注意
----
1. 不要在输入FOOOF前把PSD转成dB或log10。
2. 该脚本主要用于理解完整流程，不代表正式研究的最终参数。
3. 正式分析时应先检查代表性频谱、拟合图、R²、误差及异常峰。
4. 通道平均结果适合教学观察；正式研究通常还需根据研究问题决定
   使用通道级、脑区级、源空间或被试级分析。
"""

    (output_dir / "README_输出说明.txt").write_text(
        readme,
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取EEGLAB .set，计算Welch PSD并用FOOOF 1.1.1参数化。"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="预处理好的EEGLAB .set文件路径；省略时弹出文件选择窗口。",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="输出文件夹；省略时在输入文件旁创建 *_fooof_demo 文件夹。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    set_path = Path(args.input).expanduser() if args.input else choose_set_file()
    set_path = set_path.resolve()

    if not set_path.exists():
        raise FileNotFoundError(f"文件不存在：{set_path}")

    if set_path.suffix.lower() != ".set":
        raise ValueError(f"请选择.set文件，当前文件为：{set_path.name}")

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = set_path.parent / f"{set_path.stem}_fooof_demo"

    print("=" * 60)
    print("EEGLAB .set → Welch PSD → FOOOF 1.1.1 教学演示")
    print("=" * 60)
    print(f"输入文件：{set_path}")
    print(f"输出目录：{output_dir}")

    print("\n[1/4] 正在读取.set文件……")
    data, sfreq, channel_names, data_kind, extra_info = load_eeglab_set(
        set_path
    )
    print(f"数据类型：{data_kind}")
    print(f"采样率：{sfreq:.3f} Hz")
    print(f"EEG通道数：{len(channel_names)}")
    print(f"数据形状：{data.shape}")

    print("\n[2/4] 正在计算Welch功率谱……")
    freqs, channel_psd, mean_psd, welch_info = compute_welch_psd(
        data,
        sfreq,
    )
    print(
        "PSD频率范围："
        f"{freqs[0]:.3f}–{freqs[-1]:.3f} Hz"
    )
    print(
        "频率分辨率："
        f"{welch_info['frequency_resolution_hz']:.6f} Hz"
    )

    print("\n[3/4] 正在拟合通道平均功率谱……")
    mean_model, effective_fit_range = fit_one_spectrum(
        freqs,
        mean_psd,
    )
    print(
        "平均频谱非周期参数："
        f"{np.asarray(mean_model.aperiodic_params_)}"
    )
    print(f"平均频谱检测到峰数：{mean_model.n_peaks_}")
    print(f"平均频谱R²：{mean_model.r_squared_:.4f}")
    print(f"平均频谱误差：{mean_model.error_:.4f}")

    print("\n[4/4] 正在保存结果……")
    save_outputs(
        set_path=set_path,
        output_dir=output_dir,
        freqs=freqs,
        channel_psd=channel_psd,
        mean_psd=mean_psd,
        channel_names=channel_names,
        mean_model=mean_model,
        effective_fit_range=effective_fit_range,
        sfreq=sfreq,
        data_kind=data_kind,
        extra_info=extra_info,
        welch_info=welch_info,
    )

    print("\n分析完成。")
    print(f"结果位置：{output_dir}")
    print("建议先查看：05_mean_spectrum_fooof_fit.png")
    print("再查看：03_aperiodic_parameters.csv 和 04_periodic_peak_parameters.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中止运行。")
        raise SystemExit(130)
    except Exception as exc:
        print("\n运行失败：")
        print(exc)
        raise SystemExit(1)
