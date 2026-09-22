#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""检查教学包依赖，并用一条合成频谱完成一次最小FOOOF拟合。"""

from __future__ import annotations

import sys
import numpy as np
import scipy
import matplotlib
import mne
import fooof
from fooof import FOOOF


def main() -> int:
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    print("SciPy:", scipy.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("MNE:", mne.__version__)
    print("FOOOF:", fooof.__version__)

    freqs = np.arange(2.0, 40.01, 0.25)
    log_power = 1.0 - 1.4 * np.log10(freqs)
    log_power += 0.35 * np.exp(-0.5 * ((freqs - 10.0) / 1.2) ** 2)
    power = 10 ** log_power

    model = FOOOF(
        peak_width_limits=[1.0, 8.0],
        max_n_peaks=4,
        min_peak_height=0.05,
        peak_threshold=2.0,
        aperiodic_mode="fixed",
        verbose=False,
    )
    model.fit(freqs, power, [2, 40])

    if not model.has_model:
        print("[FAILED] FOOOF did not produce a model.")
        return 1

    print("[SUCCESS] Environment and FOOOF fitting are working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
