# SpecParam / FOOOF 教学演示包

[English](README.md)

通过内置合成功率谱和 EEGLAB .set 数据，学习周期峰与非周期背景的频谱参数化。实际依赖为 FOOOF 1.1.1。

## 开始使用

准备 64 位 Python 3.11（说明支持 3.9–3.12），依次运行 `01_install.bat`、`03_run_builtin_demo.bat`；内置示例成功后，再运行 `02_run_set_demo.bat` 选择自己的数据。详见 [完整使用说明](README_请先看.txt)。

## 适用范围

教学版使用全部 EEG 通道的平均 PSD，默认拟合 2–40 Hz、fixed 非周期模型。用于理解流程，不代表正式研究的最终分析方案。

本次上传保留原有工具文件，检查文件一致性及适用的 Python 语法；不将上传检查等同于科学有效性验证或完整应用测试。
