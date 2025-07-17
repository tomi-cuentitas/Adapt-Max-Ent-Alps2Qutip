import pickle
from pathlib import Path
from typing import Sequence, Any

import numpy as np
import pandas as pd


def ensure_dir(directory):
    if isinstance(directory, np.ndarray):
        raise TypeError("Path was overwritten as ndarray. Check for variable overwrite.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_checkpoint(
    sim_id,
    times,
    evs,
    evs_exact=None,
    out_dir=None,
    step=None,
    orth_lens=None,
    ints_err=None,  # ✅ new argument for instantaneous error
):
    fname = Path(out_dir) / f"traj_{sim_id}_step{step if step is not None else 'final'}.csv"

    data = {
        "t": times,
        "ev": np.asarray(evs, dtype=np.complex128),
    }

    if evs_exact is not None:
        data["ev_exact"] = np.asarray(evs_exact, dtype=np.complex128)

    #if orth_lens is not None:
    #    data["orth_len"] = np.asarray(orth_lens, dtype=np.float64)

    #if ints_err is not None:
    #    data["ints_err"] = np.asarray(ints_err, dtype=np.complex128)

    df = pd.DataFrame(data)
    df.to_csv(fname, index=False)
