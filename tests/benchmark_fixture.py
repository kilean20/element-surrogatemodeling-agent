"""Generated algebraic regression controls; no facility data or pretrained assets."""

from pathlib import Path

import numpy as np

from element_surrogate.benchmark.protocol import SCHEMA, file_hash, protocol_identity, write_json

ADAPTER = """import numpy as np

class Model:
    def __init__(self, x, y, configuration, seed):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, .2, (x.shape[1], configuration["width"]))
        self.b1 = np.zeros(configuration["width"])
        self.w2 = np.zeros((configuration["width"], y.shape[1]))
        self.b2 = np.zeros(y.shape[1])
        self.x, self.residual = x, y-x[:, :y.shape[1]]
        self.configuration = configuration
        self.steps = 0
    def step(self):
        self.steps += 1
        if self.configuration.get("slow"):
            import time
            time.sleep(5)
        h = np.tanh(self.x@self.w1+self.b1)
        error = (h@self.w2+self.b2-self.residual)*2/len(self.x)
        if self.configuration.get("conserved_index") is not None:
            error[:, self.configuration["conserved_index"]] = 0
        hidden = (error@self.w2.T)*(1-h*h)
        rate = self.configuration["learning_rate"]
        self.w2 -= rate*h.T@error
        self.b2 -= rate*error.sum(0)
        self.w1 -= rate*self.x.T@hidden
        self.b1 -= rate*hidden.sum(0)
    def predict(self, x):
        correction = np.tanh(x@self.w1+self.b1)@self.w2+self.b2
        if self.configuration.get("conserved_index") is not None:
            correction[:, self.configuration["conserved_index"]] = 0
        return x[:, :self.w2.shape[1]]+correction
    def save(self, path):
        with open(path, "wb") as stream:
            np.savez(stream, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

def build_adapter(x, y, configuration, seed):
    return Model(x, y, configuration, seed)

def load_adapter(path, configuration):
    with np.load(path, allow_pickle=False) as data:
        model = object.__new__(Model)
        for key in ("w1", "b1", "w2", "b2"):
            setattr(model, key, data[key])
    model.configuration = configuration
    return model
"""


def approval(directory, protocol, gate):
    path = Path(directory) / f"approval_{gate}.json"
    write_json(
        path,
        {
            "schema": "element_surrogate.approval.v1",
            "protocol_sha256": protocol_identity(protocol),
            "gate": gate,
            "decision": "approved",
            "actor": "synthetic-test-user",
            "user_statement": "Synthetic fixture approval; not a real institutional decision.",
        },
    )
    return str(path)


def make_case(directory, dimension=4, *, configurations=2):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    adapter = directory / "adapter.py"
    adapter.write_text(ADAPTER)
    coordinates = [
        {
            "name": "x",
            "unit": "mm",
            "role": "position",
            "definition": "horizontal position",
            "error_scale": 0.1,
        },
        {
            "name": "xp",
            "unit": "mrad",
            "role": "slope",
            "definition": "horizontal direction ratio",
            "error_scale": 0.1,
        },
        {
            "name": "y",
            "unit": "mm",
            "role": "position",
            "definition": "vertical position",
            "error_scale": 0.1,
        },
        {
            "name": "yp",
            "unit": "mrad",
            "role": "slope",
            "definition": "vertical direction ratio",
            "error_scale": 0.1,
        },
    ]
    if dimension == 6:
        coordinates.append(
            {
                "name": "path",
                "unit": "mm",
                "role": "arrival",
                "definition": "relative path increment",
                "error_scale": 0.01,
            }
        )
    if dimension >= 5:
        coordinates.append(
            {
                "name": "delta",
                "unit": "1",
                "role": "energy",
                "definition": "relative momentum deviation",
                "error_scale": 0.001,
            }
        )
    inputs = [c["name"] for c in coordinates] + ["control"]
    identity = "1" * 64
    splits = {}
    for i, (split, rows) in enumerate((("train", 60), ("validation", 31), ("test", 23))):
        x = np.random.default_rng(100 + i).uniform(-0.3, 0.3, (rows, dimension + 1))
        y = x[:, :dimension].copy()
        y[:, 0] += 0.2 * x[:, 1] + 0.03 * x[:, 0] ** 3 + 0.05 * x[:, -1]
        y[:, 1] -= 0.1 * x[:, 0]
        y[:, 2] += 0.2 * x[:, 3]
        y[:, 3] += 0.1 * x[:, 2]
        if dimension == 6:
            y[:, 4] += 0.03 * (x[:, 1] ** 2 + x[:, 3] ** 2) + 0.07 * x[:, 5]
        groups = [f"{split}_a", f"{split}_b"]
        settings = np.array([groups[0]] * (rows - 2) + [groups[1]] * 2)
        path = directory / f"{split}.npz"
        np.savez_compressed(
            path,
            inputs=x,
            targets=y,
            setting_id=settings,
            sample_id=np.arange(rows, dtype=np.uint64) + np.uint64(2**60 + i * 1000),
            input_names=inputs,
            input_units=[c["unit"] for c in coordinates] + ["1"],
            dataset_identity_sha256=identity,
            coordinate_names=[c["name"] for c in coordinates],
            coordinate_units=[c["unit"] for c in coordinates],
        )
        splits[split] = {
            "path": path.name,
            "sha256": file_hash(path),
            "rows": rows,
            "groups": groups,
        }
    candidates = []
    for name, width in (("narrow", 4), ("wide", 8))[:configurations]:
        candidates.append(
            {
                "id": name,
                "architecture": "residual_mlp",
                "kind": "stochastic",
                "seeds": [12, 19],
                "configuration": {
                    "width": width,
                    "learning_rate": 0.2,
                    "conserved_index": dimension - 1 if dimension >= 5 else None,
                },
                "rationale": "bounded algebraic neural-regression control",
                "forward_description": "Identity plus a trained tanh hidden layer and "
                "linear correction; momentum deviation passes through exactly when present.",
                "max_steps": 35,
                "max_wall_seconds": 20,
            }
        )
    p = {
        "schema": SCHEMA,
        "name": "synthetic-element-map",
        "coordinates": coordinates,
        "coordinate_convention": "Manufactured entrance/exit plane-coordinate regression control.",
        "inputs": inputs,
        "input_units": [c["unit"] for c in coordinates] + ["1"],
        "forbidden_inputs": ["target", "path_maximum"],
        "physics": {
            "field_sha256": "2" * 64,
            "reference_sha256": "3" * 64,
            "reconstruction_sha256": "4" * 64,
            "integration_qualification_sha256": "5" * 64,
            "region_sha256": "6" * 64,
            "native_planes": True,
            "qualification_scope": "Manufactured regression control only; "
            "no physical field or machine qualification is asserted.",
            "boundary_convention": "Fixed native plane coordinates; "
            "no effective-length transformation.",
        },
        "dataset": {
            "identity_sha256": identity,
            "access_and_reuse": "Generated synthetic test data only.",
            "splits": splits,
        },
        "resources": {
            "max_rows_per_split": 100,
            "max_npz_uncompressed_bytes": 100000,
            "max_checkpoint_bytes": 100000,
            "max_study_wall_seconds": 100,
            "max_test_wall_seconds": 30,
        },
        "candidates": candidates,
        "selection": {
            "primary": "worst_scaled_setting_mean_rmse",
            "relative_accuracy_tie": 0.01,
            "incomplete_policy": "refuse",
        },
        "timing": {
            "batch_size": 16,
            "repetitions": 3,
            "warmup_calls": 1,
            "threads": 1,
            "input_dtype": "float64",
            "scope": "adapter_predict_call",
            "hardware": "test host",
            "device": "cpu",
            "synchronization": "synchronous_cpu",
            "golden_atol": [1e-12] * dimension,
            "included_operations": "full NumPy input-to-output prediction, "
            "identity base and correction",
            "excluded_operations": "model loading, fitting and data-file reads",
        },
        "reproduction": {
            "adapter_path": adapter.name,
            "adapter_sha256": file_hash(adapter),
            "software": "Python with NumPy; exact worker versions are recorded",
        },
    }
    path = directory / "protocol.json"
    write_json(path, p)
    approvals = [
        approval(directory, p, gate)
        for gate in ("description_target", "region_ranges", "compute_budget")
    ]
    return path, p, approvals
