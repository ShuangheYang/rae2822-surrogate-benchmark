"""Small CPU numerical checks. Does not train on or evaluate the CFD dataset."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))
os.environ["AIRFOIL_DEVICE"] = "cpu"
import numpy as np
from common import metrics, with_ratio, read_json
from mlp_engine import fit, predict, save_model, reload_model
from kriging_engine import ExactIsotropicKriging


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(48, 20))
    y = np.column_stack(
        (
            0.8 + 0.02 * x[:, 0] + 0.003 * x[:, 1] ** 2,
            0.02 + 0.0003 * x[:, 2] + 0.0001 * x[:, 3] ** 2,
        )
    )
    config = copy.deepcopy(
        read_json(PROJECT / "configs/protocol.json")["candidates"][3]
    )
    config.update(
        epochs=2, minimum_epochs=0, patience=10, validation_interval=1, lbfgs_steps=5
    )
    net, meta = fit(x[:32], y[:32], x[32:], y[32:], config, 11)
    assert meta["device"] == "cpu"
    np.testing.assert_allclose(meta["x_mean"], x[:32].mean(0), atol=1e-15)
    pred = predict(net, x[32:], meta)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = root / "mlp.npz"
        save_model(meta, path)
        loaded, lmeta = reload_model(path, "cpu")
        np.testing.assert_allclose(
            predict(loaded, x[32:], lmeta), pred, rtol=1e-12, atol=1e-14
        )
        gp = ExactIsotropicKriging(
            {"device": "cpu", "optimize": False, "verbose": False}
        ).fit(x[:32], y[:32])
        gp.save(root / "gp.npz")
        back = ExactIsotropicKriging.load(root / "gp.npz", device="cpu")
        np.testing.assert_allclose(
            back.predict(x[32:]), gp.predict(x[32:]), rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(with_ratio(pred)[:, 2], pred[:, 0] / pred[:, 1])
        assert metrics(y[32:], pred)["E2_percent"] >= 0
        # Driver resume verifies data, code, configuration and model bytes.
        import run_mlp_study as driver

        driver.ROOT = root
        (root / "data").mkdir()
        np.savez(
            root / "data/development.npz",
            X=x[:32],
            Y=y[:32],
            X_val=x[32:],
            Y_val=y[32:],
            ids=np.array([f"{i:05d}" for i in range(32)]),
        )
        (root / "protocol.json").write_text(json.dumps({"test": "synthetic"}))
        driver.development_data = lambda: np.load(root / "data/development.npz")
        result = driver.one(config, 11, np.arange(32), "smoke")
        assert not Path(result["model_path"]).is_absolute()
        assert driver.one(config, 11, np.arange(32), "smoke") == result
        mismatch = copy.deepcopy(config)
        mismatch["learning_rate"] *= 2
        try:
            driver.one(mismatch, 11, np.arange(32), "smoke")
        except RuntimeError:
            pass
        else:
            raise AssertionError("A different configuration was incorrectly resumed")
    print(
        "PASS: CPU optimization, train-only scaling, NPZ round trips, ratio, metrics, resume checks."
    )


if __name__ == "__main__":
    main()
