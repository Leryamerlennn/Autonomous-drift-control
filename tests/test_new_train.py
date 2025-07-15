# pylint: disable=duplicate-code
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ML'))
import new_train as nt  # pylint: disable=wrong-import-position

def test_load_csvs(tmp_path):
    df1 = pd.DataFrame({'t_sec':[0], 'x_world':[0], 'y_world':[0], 'yaw_rad':[0]})
    df2 = pd.DataFrame({'t_sec':[1], 'x_world':[1], 'y_world':[0], 'yaw_rad':[0]})
    p1 = tmp_path / 'a.csv'
    p2 = tmp_path / 'b.csv'
    df1.to_csv(p1, index=False)
    df2.to_csv(p2, index=False)
    loaded = nt.load_csvs(str(tmp_path / '*.csv'))
    assert len(loaded) == 2

def test_add_speed_beta():
    df = pd.DataFrame({'t_sec':[0.0,1.0],
                       'x_world':[0.0,1.0],
                       'y_world':[0.0,0.0],
                       'yaw_rad':[0.0,0.0]})
    out = nt.add_speed_beta(df)
    assert {'vx','vy','speed','beta'} <= set(out.columns)
    assert np.isclose(out.loc[0,'vx'],1.0)
    assert np.isclose(out.loc[0,'vy'],0.0)

def test_build_matrices():
    df = pd.DataFrame({
        'yawRate':[0.0,0.1,0.2],
        'ay_world':[0.0,0.0,0.0],
        'beta':[0.0,0.0,0.0],
        'speed':[0.0,0.5,1.0],
        'steer':[nt.STEER_C, nt.STEER_C, nt.STEER_C],
        'gas':[nt.GAS_C, nt.GAS_C, nt.GAS_C]
    })
    x_mat, delta_s = nt.build_matrices(df)
    assert x_mat.shape == (2, 6)
    assert delta_s.shape == (2, 4)

def test_load_dataset_npz(tmp_path):
    npz_path = tmp_path / 'data.npz'
    np.savez(npz_path, X=np.zeros((2,6)), Y=np.ones((2,4)),
             mu_X=np.zeros(6), sig_X=np.ones(6),
             mu_Y=np.zeros(4), sig_Y=np.ones(4))
    x_mat, y_mat, mu_x, _sig_x, _mu_y, _sig_y = nt.load_dataset(str(npz_path))
    assert x_mat.shape == (2, 6)
    assert y_mat.shape == (2, 4)
    assert np.allclose(mu_x, np.zeros(6))

def test_dynnet_forward():
    net = nt.DynNet()
    out = net(torch.zeros(1,6))
    assert out.shape == (1,4)
