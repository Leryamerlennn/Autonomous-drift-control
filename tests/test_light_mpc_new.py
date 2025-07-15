# pylint: disable=duplicate-code
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import serial

torch = pytest.importorskip("torch")

# pylint: disable=protected-access


def load_module(path: Path):
    """Load light mpc module without running its endless loop."""
    class StopSerial:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            raise RuntimeError("stop")
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    serial.Serial = StopSerial
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("light_mpc", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[arg-type]
    except RuntimeError:
        pass
    return module

# path to module
MOD_PATH = Path(__file__).resolve().parents[1] / 'ML' / 'light_version' / 'mpc_new.py'
mpc = load_module(MOD_PATH)


def test_fix_state_dict_keys():
    src = {
        'net.0.weight': 1,
        'net.2.bias': 2,
        'net.4.weight': 3,
        'other': 4,
    }
    res = mpc._fix_state_dict_keys(src)
    assert set(res.keys()) == {
        'net.0.fc1.weight',
        'net.2.fc2.bias',
        'net.4.fc3.weight',
        'other',
    }


def test_norm_denorm_roundtrip():
    x = np.array([1.0, 2.0, 3.0])
    mu = np.array([0.5, 1.5, 2.5])
    sig = np.array([0.5, 0.5, 0.5])
    n = mpc._norm(x, mu, sig)
    d = mpc._denorm(torch.tensor(n), mu, sig)
    assert torch.allclose(d, torch.tensor(x))


def test_mpc_control_basic(monkeypatch):

    class Dummy(torch.nn.Module):
        """Minimal network for testing."""
        # pylint: disable=too-few-public-methods
        def forward(self, _x):
            return torch.zeros(4)
    dummy = Dummy()
    monkeypatch.setattr(mpc, 'MODEL', dummy)
    monkeypatch.setattr(mpc, 'N_SAMPLES', 1)
    monkeypatch.setattr(mpc, 'HORIZON', 1)
    monkeypatch.setattr(mpc, 'SIGMA', 0.0)
    monkeypatch.setattr(mpc, 'MU_S', np.zeros(6))
    monkeypatch.setattr(mpc, 'SIG_S', np.ones(6))
    monkeypatch.setattr(mpc, 'MU_A', np.zeros(2))
    monkeypatch.setattr(mpc, 'SIG_A', np.ones(2))
    monkeypatch.setattr(mpc, 'MU_SN', np.zeros(4))
    monkeypatch.setattr(mpc, 'SIG_SN', np.ones(4))
    monkeypatch.setattr(mpc, 'STEER_SP', 1)
    monkeypatch.setattr(mpc, 'STEER_C', 0)
    monkeypatch.setattr(mpc, 'GAS_SP', 1)
    monkeypatch.setattr(mpc, 'GAS_C', 0)
    monkeypatch.setattr(mpc, 'STEER_MIN', -1)
    monkeypatch.setattr(mpc, 'STEER_MAX', 1)
    monkeypatch.setattr(mpc, 'GAS_MIN', -1)
    monkeypatch.setattr(mpc, 'GAS_MAX', 1)

    steer, gas = mpc.mpc_control(np.zeros(6))
    assert steer == 0
    assert gas == 0
