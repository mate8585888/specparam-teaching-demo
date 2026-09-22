#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEGLAB .set → Welch PSD → FOOOF 频谱参数化（极简教学版）

用户操作：
1. 双击 02_run_set_demo.bat；
2. 在窗口中选择预处理好的 .set 文件；
3. 程序自动输出平均PSD、非周期参数、周期峰参数和拟合图。

本脚本有意只分析“全部EEG通道的平均功率谱”，用于第一次理解流程。
正式研究通常还需开展通道级、脑区级或源空间分析。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import os
import sys
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import mne
from fooof import FOOOF


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# ---------------- 教学演示参数 ----------------
PSD_RANGE = (1.0, 40.0)
FIT_RANGE = (2.0, 40.0)
WELCH_WINDOW_SECONDS = 4.0

PEAK_WIDTH_LIMITS = (1.0, 8.0)
MAX_N_PEAKS = 6
MIN_PEAK_HEIGHT = 0.10
PEAK_THRESHOLD = 2.0
APERIODIC_MODE = "fixed"


def show_message(title: str, message: str, error: bool = False) -> None:
    """优先使用Windows消息框；失败时打印到终端。"""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        if error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)
        root.destroy()
    except Exception:
        print(f"\n{title}\n{message}")


def choose_set_file() -> Path:
    """弹出文件选择窗口；无Tkinter时允许用户粘贴路径。"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="请选择预处理好的 EEGLAB .set 文件",
            filetypes=[("EEGLAB SET", "*.set"), ("所有文件", "*.*")],
        )
        root.destroy()

        if not selected:
            raise SystemExit("用户取消了文件选择。")
        return Path(selected)

    except ImportError:
        print("当前Python没有Tkinter，请粘贴.set文件完整路径：")
        return Path(input("> ").strip().strip('"'))


def open_folder(path: Path) -> None:
    if os.name == "nt":
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except OSError:
            pass


def select_eeg_indices(info: mne.Info) -> np.ndarray:
    """选择EEG通道，并排除标记为bad的通道。"""
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
    if len(picks):
        return np.asarray(picks, dtype=int)

    warnings.warn(
        "文件没有正确标记EEG通道，将排除常见辅助通道后使用其余通道。"
    )
    excluded_words = (
        "EOG", "HEOG", "VEOG", "ECG", "EKG", "EMG",
        "TRIG", "STATUS", "EVENT", "STIM", "MARK",
    )
    bads = set(info.get("bads", []))
    fallback = [
        i for i, name in enumerate(info["ch_names"])
        if name not in bads
        and not any(word in name.upper() for word in excluded_words)
    ]
    if not fallback:
        raise RuntimeError("没有找到可用于分析的EEG通道。")
    return np.asarray(fallback, dtype=int)


def read_eeglab_epoch_metadata(path: Path) -> tuple[int, int]:
    """
    读取EEGLAB分段数据的trials和pnts。

    这里只读取元数据，用于在EEG.event或EEG.epoch缺失/不完整时，
    为MNE构造占位事件。频谱分析不依赖这些占位事件的标签。
    """
    try:
        from mne.io.eeglab.eeglab import _check_load_mat
    except ImportError as exc:
        raise RuntimeError("当前MNE版本无法访问EEGLAB元数据读取器。") from exc

    try:
        # 新版MNE包含preload参数
        eeg = _check_load_mat(str(path), None, preload=False)
    except TypeError:
        # 兼容MNE 1.7等旧版签名
        eeg = _check_load_mat(str(path), None)

    trials = int(eeg.trials)
    pnts = int(eeg.pnts)

    if trials <= 1 or pnts <= 0:
        raise RuntimeError(
            f"EEGLAB元数据显示 trials={trials}, pnts={pnts}，"
            "无法按分段数据构造占位事件。"
        )

    return trials, pnts


def read_epoched_with_dummy_events(path: Path):
    """
    当EEGLAB的event/epoch结构缺失或不完整时，使用占位事件读取数据。

    占位事件只负责让MNE识别每一个epoch，不改变EEG数值，
    也不参与后续Welch PSD和FOOOF拟合。
    """
    trials, pnts = read_eeglab_epoch_metadata(path)

    events = np.zeros((trials, 3), dtype=int)
    events[:, 0] = np.arange(trials, dtype=int) * pnts
    events[:, 2] = 1

    epochs = mne.read_epochs_eeglab(
        str(path),
        events=events,
        event_id={"resting_epoch": 1},
        verbose="ERROR",
    )
    return epochs


def load_set(path: Path) -> tuple[np.ndarray, float, list[str], str]:
    """
    返回：
    连续数据：[channel, time]
    分段数据：[epoch, channel, time]
    """
    continuous_error: Exception | None = None

    try:
        raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
        picks = select_eeg_indices(raw.info)
        try:
            data = raw.get_data(picks=picks, reject_by_annotation="omit")
        except TypeError:
            data = raw.get_data(picks=picks)

        if data.size == 0:
            raise RuntimeError("连续数据为空。")

        names = [raw.ch_names[i] for i in picks]
        return (
            np.asarray(data, dtype=np.float64),
            float(raw.info["sfreq"]),
            names,
            "continuous",
        )
    except Exception as exc:
        continuous_error = exc

    standard_epoch_error: Exception | None = None

    try:
        epochs = mne.read_epochs_eeglab(str(path), verbose="ERROR")
        loader_mode = "epoched_with_original_events"
    except Exception as exc:
        standard_epoch_error = exc

        # 静息态数据经常已切成epoch，但EEG.event或EEG.epoch为空/不完整。
        # MNE默认会尝试从这些字段构建事件，可能出现list index out of range。
        # 对PSD而言事件标签不是必要信息，因此改用占位事件继续读取。
        try:
            print(
                "检测到分段数据的事件结构无法由MNE直接解析，"
                "正在使用占位事件兼容读取……"
            )
            epochs = read_epoched_with_dummy_events(path)
            loader_mode = "epoched_with_generated_placeholder_events"
        except Exception as fallback_error:
            raise RuntimeError(
                "无法读取该.set文件。\n\n"
                f"按连续数据读取时：{continuous_error}\n\n"
                f"按原始事件读取分段数据时：{standard_epoch_error}\n\n"
                f"使用占位事件兼容读取时：{fallback_error}\n\n"
                "请确认：\n"
                "1. 若存在同名.fdt文件，它与.set位于同一文件夹；\n"
                "2. 文件能在EEGLAB中正常打开；\n"
                "3. 文件路径和文件名没有被移动或修改。\n\n"
                "若文件可在EEGLAB中打开，请上传.set、配套.fdt和错误日志进一步检查。"
            ) from fallback_error

    picks = select_eeg_indices(epochs.info)
    data = epochs.get_data(picks=picks)

    if data.size == 0:
        raise RuntimeError("分段数据为空。")

    names = [epochs.ch_names[i] for i in picks]
    return (
        np.asarray(data, dtype=np.float64),
        float(epochs.info["sfreq"]),
        names,
        loader_mode,
    )


def calculate_mean_psd(
    data: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    对每通道计算Welch PSD，再在线性功率尺度下：
    先平均epochs（如有），再平均通道。
    """
    n_times = int(data.shape[-1])
    nperseg = min(int(round(WELCH_WINDOW_SECONDS * sfreq)), n_times)

    if nperseg < 16:
        raise RuntimeError("有效数据太短，无法计算稳定的功率谱。")

    noverlap = nperseg // 2

    freqs, psd = welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend="constant",
        scaling="density",
        axis=-1,
        average="mean",
    )

    # 分段数据：[epoch, channel, frequency] → 先平均epoch
    if data.ndim == 3:
        psd = np.mean(psd, axis=0)

    # 现在是：[channel, frequency] → 平均通道
    mean_psd = np.mean(psd, axis=0)

    highest = min(PSD_RANGE[1], sfreq / 2.0)
    mask = (freqs >= PSD_RANGE[0]) & (freqs <= highest)
    freqs = np.asarray(freqs[mask], dtype=float)
    mean_psd = np.asarray(mean_psd[mask], dtype=float)

    if freqs.size < 5:
        raise RuntimeError("可用频率点过少，请检查采样率和数据长度。")

    if not np.all(np.isfinite(mean_psd)):
        raise RuntimeError("功率谱出现NaN或无穷值。")

    # FOOOF要求功率为正，并且必须是线性功率而非dB
    mean_psd = np.maximum(mean_psd, np.finfo(float).tiny)
    resolution = float(freqs[1] - freqs[0])

    return freqs, mean_psd, resolution


def fit_fooof(
    freqs: np.ndarray,
    spectrum: np.ndarray,
) -> tuple[FOOOF, tuple[float, float]]:
    fit_high = min(FIT_RANGE[1], float(freqs[-1]))
    fit_range = (FIT_RANGE[0], fit_high)

    if fit_range[1] <= fit_range[0]:
        raise RuntimeError("数据的最高可用频率不足以完成FOOOF拟合。")

    model = FOOOF(
        peak_width_limits=PEAK_WIDTH_LIMITS,
        max_n_peaks=MAX_N_PEAKS,
        min_peak_height=MIN_PEAK_HEIGHT,
        peak_threshold=PEAK_THRESHOLD,
        aperiodic_mode=APERIODIC_MODE,
        verbose=False,
    )
    model.fit(freqs, spectrum, list(fit_range))

    if not model.has_model:
        raise RuntimeError("FOOOF没有返回有效模型。")

    return model, fit_range


def save_results(
    input_path: Path,
    data_kind: str,
    sfreq: float,
    channel_names: list[str],
    freqs: np.ndarray,
    spectrum: np.ndarray,
    frequency_resolution: float,
    model: FOOOF,
    fit_range: tuple[float, float],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = input_path.parent / f"{input_path.stem}_fooof教学结果_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. FOOOF实际输入：线性尺度PSD
    np.savetxt(
        out_dir / "01_FOOOF输入_平均PSD.csv",
        np.column_stack([freqs, spectrum]),
        delimiter=",",
        header="frequency_hz,power_spectral_density_linear",
        comments="",
        fmt="%.12e",
    )

    # 2. 非周期参数
    ap = np.asarray(model.aperiodic_params_, dtype=float)
    with (out_dir / "02_非周期参数.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["offset", "exponent", "r_squared", "error", "number_of_peaks"]
        )
        writer.writerow(
            [ap[0], ap[1], model.r_squared_, model.error_, model.n_peaks_]
        )

    # 3. 周期峰参数
    with (out_dir / "03_周期峰参数.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["peak_index", "center_frequency_hz",
             "peak_power_above_aperiodic_log10", "bandwidth_hz"]
        )
        for index, peak in enumerate(model.peak_params_, start=1):
            writer.writerow([index, peak[0], peak[1], peak[2]])

    # 4. 频谱模型曲线
    curves = np.column_stack(
        [
            model.freqs,
            model.get_data("full", "log"),
            model.get_model("aperiodic", "log"),
            model.get_model("peak", "log"),
            model.get_model("full", "log"),
        ]
    )
    np.savetxt(
        out_dir / "04_模型曲线.csv",
        curves,
        delimiter=",",
        header=(
            "frequency_hz,original_log10_psd,"
            "aperiodic_fit_log10,periodic_component_log10,"
            "full_model_fit_log10"
        ),
        comments="",
        fmt="%.12e",
    )

    # 5. 拟合图
    model.plot(plot_peaks="shade", plot_aperiodic=True, add_legend=True)
    plt.title(f"FOOOF fit: {input_path.name} (mean across EEG channels)")
    plt.tight_layout()
    plt.savefig(out_dir / "05_频谱参数化拟合图.png", dpi=220, bbox_inches="tight")
    plt.close("all")

    # 6. 中文说明
    peaks_text = (
        "\n".join(
            [
                f"峰{i}: CF={peak[0]:.3f} Hz, "
                f"PW={peak[1]:.6f}, BW={peak[2]:.3f} Hz"
                for i, peak in enumerate(model.peak_params_, start=1)
            ]
        )
        if model.n_peaks_
        else "未检测到满足当前阈值的周期峰。"
    )

    summary = f"""SpecParam / FOOOF 教学分析结果
================================

输入文件：
{input_path}

本次分析做了什么：
预处理后的.set
→ Welch方法计算PSD
→ 在“线性功率尺度”下平均EEG通道
→ 输入FOOOF 1.1.1
→ 拟合非周期背景与周期峰

数据概况：
数据读取方式：{data_kind}
采样率：{sfreq:.3f} Hz
纳入EEG通道数：{len(channel_names)}
通道：{", ".join(channel_names)}
频率分辨率：{frequency_resolution:.6f} Hz
拟合范围：{fit_range[0]:.1f}–{fit_range[1]:.1f} Hz

非周期参数：
offset = {ap[0]:.6f}
exponent = {ap[1]:.6f}

周期峰：
{peaks_text}

拟合质量：
R² = {model.r_squared_:.6f}
error = {model.error_:.6f}

重要说明：
1. FOOOF输入的是功率谱，不是原始EEG时间序列。
2. 输入功率必须保持在线性尺度，不能预先转换成dB或log10。
3. 本脚本分析的是“所有EEG通道的平均PSD”，主要用于教学。
4. 当前参数只用于第一次试跑，不能直接作为正式论文的统一参数。
5. 正式研究还要检查频谱形状、拟合图、异常峰、通道差异和参数稳定性。
"""
    (out_dir / "README_结果说明.txt").write_text(summary, encoding="utf-8")
    return out_dir


def main() -> int:
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1].strip('"'))
    else:
        input_path = choose_set_file()

    input_path = input_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在：{input_path}")
    if input_path.suffix.lower() != ".set":
        raise ValueError("请选择EEGLAB的.set文件。")

    print("=" * 64)
    print("EEGLAB .set → Welch PSD → FOOOF 教学演示")
    print("=" * 64)
    print("输入：", input_path)

    print("\n[1/4] 读取.set数据……")
    data, sfreq, channel_names, data_kind = load_set(input_path)
    print("数据类型：", data_kind)
    print("数据形状：", data.shape)
    print("采样率：", sfreq)
    print("EEG通道数：", len(channel_names))

    print("\n[2/4] 计算Welch功率谱……")
    freqs, spectrum, resolution = calculate_mean_psd(data, sfreq)
    print(f"频率范围：{freqs[0]:.3f}–{freqs[-1]:.3f} Hz")
    print(f"频率分辨率：{resolution:.6f} Hz")

    print("\n[3/4] 运行FOOOF拟合……")
    model, fit_range = fit_fooof(freqs, spectrum)
    print("非周期参数：", model.aperiodic_params_)
    print("周期峰数量：", model.n_peaks_)
    print(f"R²：{model.r_squared_:.4f}")
    print(f"error：{model.error_:.4f}")

    print("\n[4/4] 保存结果……")
    out_dir = save_results(
        input_path=input_path,
        data_kind=data_kind,
        sfreq=sfreq,
        channel_names=channel_names,
        freqs=freqs,
        spectrum=spectrum,
        frequency_resolution=resolution,
        model=model,
        fit_range=fit_range,
    )

    message = (
        "分析完成。\n\n"
        f"结果文件夹：\n{out_dir}\n\n"
        "建议先查看：05_频谱参数化拟合图.png"
    )
    print("\n" + message)
    show_message("FOOOF教学分析完成", message)
    open_folder(out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        error_dir = PACKAGE_ROOT / "error_logs"
        error_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = error_dir / f"error_{timestamp}.txt"
        log_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )

        message = (
            f"{exc}\n\n"
            f"完整错误日志已保存到：\n{log_path}\n\n"
            "可将这个日志文件发给技术支持或AI进行排查。"
        )
        print("\n运行失败：\n" + message)
        show_message("FOOOF教学分析失败", message, error=True)
        raise SystemExit(1)
