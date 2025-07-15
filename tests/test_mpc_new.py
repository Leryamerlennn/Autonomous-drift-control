import importlib.util
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def load_module(path: Path):
    """Load module without executing its main loop."""
    import serial
    class StopSerial:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            raise RuntimeError("stop")
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    serial.Serial = StopSerial
    spec = importlib.util.spec_from_file_location("mod", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[arg-type]
    except RuntimeError:
        pass
    return module


def test_fix_state_dict_keys_heavy():
    path = Path(__file__).resolve().parents[1] / "ML" / "mpc_new.py"
    mpc = load_module(path)
    src = {
        "net.0.weight": 1,
        "net.2.bias": 2,
        "net.4.weight": 3,
        "other": 4,
    }
    res = mpc._fix_state_dict_keys(src)
    assert set(res.keys()) == {
        "net.0.fc1.weight",
        "net.2.fc2.bias",
        "net.4.fc3.weight",
        "other",
    }


def test_norm_denorm_roundtrip_heavy():
    path = Path(__file__).resolve().parents[1] / "ML" / "mpc_new.py"
    mpc = load_module(path)
    x = np.array([1.0, 2.0, 3.0])
    mu = np.array([0.5, 1.5, 2.5])
    sig = np.array([0.5, 0.5, 0.5])
    n = mpc._norm(x, mu, sig)
    d = mpc._denorm(torch.tensor(n), mu, sig)
    assert torch.allclose(d, torch.tensor(x))

