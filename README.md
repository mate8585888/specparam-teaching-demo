# SpecParam / FOOOF Teaching Demo

[中文说明](README.zh-CN.md)

A Windows-oriented teaching package for EEG power-spectrum parameterization using FOOOF 1.1.1.

## Quick start

Install 64-bit Python 3.11 (supported range in the package: 3.9–3.12). Run `01_install.bat`, then `03_run_builtin_demo.bat` for a synthetic-spectrum example. Use `02_run_set_demo.bat` to select a preprocessed EEGLAB `.set` file.

## Scope

The teaching workflow fits the mean PSD across EEG channels, with a default 2–40 Hz range and a fixed aperiodic model. It is an introduction to the workflow, not a validated final research protocol. The package uses FOOOF 1.1.1, not a newer specparam API.

This repository preserves the supplied tool files. Upload checks cover file integrity and Python syntax where applicable; they do not establish scientific validation or end-to-end application testing.
