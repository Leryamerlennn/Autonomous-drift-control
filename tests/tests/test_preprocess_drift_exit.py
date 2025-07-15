import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ML'))
import preprocess_drift_exit as pde


def make_df():
    return pd.DataFrame({
        't_sec':[0.0,1.0,2.0,3.0],
        'x_world':[0,1,2,3],
        'y_world':[0,0,0,0],
        'yaw_rad':[0,0,0,0],
        'yawRate':[0,0,0,0],
        'ay_world':[0,0,0,0],
        'steer':[pde.STEER_C]*4,
        'gas':[pde.GAS_C]*4,
        'is_drift':[1,0,0,1]
    })


def test_add_speed_beta():
    df = make_df()
    out = pde.add_speed_beta(df)
    assert 'beta' in out.columns


def test_build_matrices():
    df = make_df()
    out = pde.add_speed_beta(df)
    X,dS = pde.build_matrices(out)
    assert X.shape[1] == 6
    assert dS.shape[1] == 4


def test_extract_segments():
    df = make_df()
    seg = pde.extract_drift_exit_segments(df)
    # Should find a segment when is_drift changes 1->0
    assert not seg.empty
