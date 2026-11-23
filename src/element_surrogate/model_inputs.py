"""Model-card input units/ranges from the actual native and optional public API."""

from copy import deepcopy

import numpy as np

from element_surrogate.benchmark.protocol import file_hash, read_json
from element_surrogate.contracts import coordinate_metadata

CONTROLS = ("effective_rigidity_tm", "field_scale")
CONTROL_DEFINITIONS = (
    "signed effective rigidity p0/q; may encode inverse uniform field-amplitude scaling; "
    "does not establish source calculation current or actual momentum for timing",
    "dimensionless uniform multiplier of the supplied field; does not establish coil current",
)
RANGE_SCOPE = (
    "Declared studied entrance/control envelope, conditional on retained supported trajectories; "
    "not a uniform accuracy, survival or physical-aperture guarantee. Min/max control ranges "
    "do not certify every value or combination between discrete studied settings."
)


def collect(transport, generation, protocol, model, *, protocol_sha256=None, selected=None):
    """Read small contracts only; never inspect field payloads or particle arrays."""
    coordinates = coordinate_metadata(transport.codec)
    names = [row["name"] for row in coordinates]
    units = [row["unit"] for row in coordinates]
    if protocol["inputs"] != names + list(CONTROLS) or protocol["input_units"] != units + [
        "T*m",
        "1",
    ]:
        raise ValueError(
            "card input protocol differs from the generated native coordinate contract"
        )
    plan = generation["plan"]
    bounds = np.asarray(plan["ranges"], dtype=float)
    if (
        bounds.shape != (len(coordinates), 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > bounds[:, 1])
    ):
        raise ValueError("finite ordered generation ranges required for every beam coordinate")
    settings = np.asarray([[row[key] for key in CONTROLS] for row in plan["settings"]], dtype=float)
    if (
        settings.ndim != 2
        or settings.shape[1] != 2
        or not len(settings)
        or not np.isfinite(settings).all()
    ):
        raise ValueError("finite nonempty setting controls required for model card")
    native_beam = [
        {**deepcopy(row), "rough_valid_range": bounds[i].tolist()}
        for i, row in enumerate(coordinates)
    ]
    native_controls = [
        {
            "name": key,
            "unit": unit,
            "rough_valid_range": [float(settings[:, i].min()), float(settings[:, i].max())],
            "definition": CONTROL_DEFINITIONS[i],
            "role": "magnetic_coupling_control",
        }
        for i, (key, unit) in enumerate(zip(CONTROLS, ("T*m", "1"), strict=True))
    ]
    report = {
        "schema": "element_surrogate.model_input_contract.v1",
        "interface": "native",
        "beam_parameter_dimensions": len(coordinates),
        "beam_coordinate_system": f"{len(coordinates)}D native boundary-plane coordinates",
        "beam_reference_frame": (
            "fixed declared native entrance/exit frames; reference is not refitted per setting"
        ),
        "beam_slope_definition": (
            "horizontal/vertical direction ratios relative to each boundary normal; "
            "the listed units specify scaling; not arctan or canonical momenta"
        ),
        "beam_parameters": deepcopy(native_beam),
        "magnet_parameters": deepcopy(native_controls),
        "model_input_dimensions": len(coordinates) + 2,
        "native_beam_parameters": native_beam,
        "native_magnet_parameters": native_controls,
        "fixed_controls": {},
        "default_controls": {},
        "input_range_scope": RANGE_SCOPE,
        "input_constraints": [
            "Beam coordinate bounds apply jointly with the full supported trajectory region "
            "and reconstruction stencil support.",
            "The native model consumes beam coordinates followed by effective_rigidity_tm "
            "and field_scale; internal coupling reduction does not change its argument count.",
            "Exact setting tuples and split membership are in the bound label-source declaration; "
            "independent control extrema do not define a certified Cartesian rectangle.",
        ],
        "studied_setting_count": len(settings),
        "effective_coupling_range": plan.get("effective_coupling_range"),
        "trajectory_region": transport.trajectory_region.as_dict(),
        "stencil_support_region": transport.support.as_dict(),
        "generation_record": protocol["dataset"].get("generation_record_path"),
        "supplied_label_declaration": protocol["dataset"].get("supplied_import_request"),
        "public_interface_evidence": None,
    }
    public = model.get("public_interface")
    if public:
        from element_surrogate.public_interface import _validate

        native = read_json(public["native_contract_path"])
        interface = read_json(public["interface_path"])
        if (
            native.get("schema") != "element_surrogate.native_model.v1"
            or interface.get("schema") != "element_surrogate.public_interface.v1"
            or protocol_sha256 is None
            or selected is None
            or native.get("protocol_sha256") != protocol_sha256
            or native.get("selected") != selected
            or native.get("inputs") != protocol["inputs"]
            or native.get("input_units") != protocol["input_units"]
            or native.get("coordinates") != protocol["coordinates"]
            or interface.get("native_contract_sha256") != file_hash(public["native_contract_path"])
        ):
            raise ValueError(
                "public card interface is not bound to this selected native model/protocol"
            )
        request = interface["request"]
        factors = _validate(native, request)
        report["interface"] = "public"
        report["beam_parameters"] = [
            {
                **row,
                "unit": request["state_units"][i],
                "definition": "Native quantity: "
                + row["definition"]
                + "; public input unit: "
                + request["state_units"][i],
                "rough_valid_range": (bounds[i] / factors[i]).tolist(),
            }
            for i, row in enumerate(native_beam)
        ]
        report["fixed_controls"] = deepcopy(request.get("fixed_controls", {}))
        report["default_controls"] = deepcopy(request.get("default_controls", {}))
        report["magnet_parameters"] = [
            {**row, "api_bounds": request["control_bounds"][row["name"]]}
            for row in native_controls
            if row["name"] not in report["fixed_controls"]
        ]
        report["model_input_dimensions"] = len(coordinates) + len(report["magnet_parameters"])
        report["public_interface_evidence"] = {
            **public,
            "interface_sha256": file_hash(public["interface_path"]),
            "native_contract_sha256": file_hash(public["native_contract_path"]),
        }
        report["input_constraints"].append(
            "Public units are converted from the same native map. Fixed controls are not "
            "variable inputs; defaults remain optional inputs. API control bounds and the "
            "studied/support domain are separate constraints."
        )
    if report["effective_coupling_range"] is not None:
        report["input_constraints"].append(
            "Effective coupling field_scale / effective_rigidity_tm must lie in the "
            "recorded qualified coupling interval."
        )
    extra_conditions = model.get("frontmatter", {}).get("input_constraints", [])
    if not isinstance(extra_conditions, list):
        raise ValueError("caller-supplied input constraints must be a list")
    for condition in extra_conditions:
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError("caller-supplied input constraints must be nonempty text")
        report["input_constraints"].append("Caller-supplied extra condition: " + condition)
    for key in (
        "beam_parameter_dimensions",
        "model_input_dimensions",
        "beam_parameters",
        "magnet_parameters",
    ):
        supplied = model.get("frontmatter", {}).get(key)
        if supplied is not None and supplied != report[key]:
            raise ValueError(
                "supplied model frontmatter conflicts with generated input contract: " + key
            )
    return report


def frontmatter(report):
    return {
        key: deepcopy(report[key])
        for key in (
            "beam_parameter_dimensions",
            "beam_coordinate_system",
            "beam_reference_frame",
            "beam_slope_definition",
            "beam_parameters",
            "magnet_parameters",
            "model_input_dimensions",
            "input_range_scope",
            "input_constraints",
        )
    }


def render(report):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    def table(beam, controls):
        rows = []
        for group, parameters in (("Beam", beam), ("Magnetic coupling", controls)):
            for row in parameters:
                limits = "[" + ", ".join(f"{v:.9g}" for v in row["rough_valid_range"]) + "]"
                extra = "; API bounds " + str(row["api_bounds"]) if "api_bounds" in row else ""
                rows.append(
                    f"| {group} | {clean(row['name'])} | {clean(row['unit'])} | {limits} | "
                    f"{clean(row['definition'] + extra)} |"
                )
        return "\n".join(
            [
                "| Input group | Parameter | Unit | Rough studied range | Meaning |",
                "|---|---|---|---|---|",
                *rows,
            ]
        )

    text = [
        "### Beam and magnetic-coupling input contract",
        "",
        f"Interface documented: **{report['interface']}**. "
        f"Beam state has {report['beam_parameter_dimensions']} coordinates; "
        f"the interface accepts {report['model_input_dimensions']} scalar input arguments "
        "including controls.",
        "",
        table(report["beam_parameters"], report["magnet_parameters"]),
        "",
        report["beam_reference_frame"] + ". " + report["beam_slope_definition"] + ".",
        "",
        report["input_range_scope"],
        "",
        *["- " + item for item in report["input_constraints"]],
        "",
        "Exact trajectory/stencil regions, setting count, coupling interval and source "
        "bindings are in model_input_contract.json.",
    ]
    if report["interface"] == "public":
        text += [
            "",
            "Fixed controls: `"
            + str(report["fixed_controls"])
            + "`. Default controls: `"
            + str(report["default_controls"])
            + "`.",
            "",
            "#### Underlying native model inputs",
            "",
            table(report["native_beam_parameters"], report["native_magnet_parameters"]),
        ]
    return "\n".join(text)
