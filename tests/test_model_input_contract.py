"""Native/public units and studied ranges stay bound to the selected map."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from element_surrogate.benchmark.protocol import read_json, write_json
from element_surrogate.contracts import StaticMagneticCodec, coordinate_metadata
from element_surrogate.model_inputs import collect, render
from element_surrogate.public_interface import export_interface


def case(dimension=4, longitudinal="path_m"):
    codec = StaticMagneticCodec(dimension, longitudinal)
    region = SimpleNamespace(as_dict=lambda: {"kind": "synthetic numerical support"})
    transport = SimpleNamespace(codec=codec, trajectory_region=region, support=region)
    coordinates = coordinate_metadata(codec)
    ranges = [[-0.02, 0.03], [-0.01, 0.01], [-0.004, 0.004], [-0.003, 0.003]]
    if dimension == 6:
        ranges.append([-1e-9, 1e-9] if longitudinal == "time_s" else [-0.001, 0.001])
    if dimension > 4:
        ranges.append([-0.05, 0.05])
    generation = {
        "plan": {
            "ranges": ranges,
            "settings": [
                {"effective_rigidity_tm": -1.2, "field_scale": -0.9},
                {"effective_rigidity_tm": 1.1, "field_scale": 1.3},
            ],
            "effective_coupling_range": [0.7, 1.3],
        }
    }
    protocol = {
        "inputs": [c["name"] for c in coordinates] + ["effective_rigidity_tm", "field_scale"],
        "input_units": [c["unit"] for c in coordinates] + ["T*m", "1"],
        "coordinates": coordinates,
        "dataset": {"generation_record_path": "synthetic/generation.json"},
    }
    return transport, generation, protocol


@pytest.mark.parametrize(
    "dimension,longitudinal",
    [(4, "path_m"), (5, "path_m"), (6, "path_m"), (6, "ct_m"), (6, "time_s")],
)
def test_all_beam_dimensions_have_actual_units_and_ranges(dimension, longitudinal):
    transport, generation, protocol = case(dimension, longitudinal)
    report = collect(transport, generation, protocol, {})
    assert report["beam_parameter_dimensions"] == dimension
    assert report["model_input_dimensions"] == dimension + 2
    assert [p["unit"] for p in report["beam_parameters"]] == protocol["input_units"][:-2]
    assert [p["rough_valid_range"] for p in report["beam_parameters"]] == generation["plan"][
        "ranges"
    ]
    assert report["magnet_parameters"][0]["rough_valid_range"] == [-1.2, 1.1]
    assert report["magnet_parameters"][1]["unit"] == "1"
    assert "discrete studied settings" in report["input_range_scope"]
    assert "coil current" in render(report)
    assert "canonical momenta" in render(report)


@pytest.mark.parametrize("key", ["inputs", "input_units"])
def test_protocol_coordinate_mismatch_is_not_hidden(key):
    transport, generation, protocol = case()
    protocol[key][0] = "wrong"
    with pytest.raises(ValueError, match="differs from"):
        collect(transport, generation, protocol, {})


def public_case(tmp_path, dimension=4, longitudinal="path_m"):
    transport, generation, protocol = case(dimension, longitudinal)
    selected = {"candidate_id": "synthetic", "seed": 1}
    native = {
        "schema": "element_surrogate.native_model.v1",
        "protocol_sha256": "a" * 64,
        "selected": selected,
        "physics": {
            "native_planes": True,
            "qualified_ranges": generation["plan"]["ranges"],
            "qualified_settings": generation["plan"]["settings"],
            "effective_coupling_range": generation["plan"]["effective_coupling_range"],
        },
        **{k: protocol[k] for k in ("inputs", "input_units", "coordinates")},
    }
    native_path = tmp_path / "contract.json"
    request_path = tmp_path / "public_request.json"
    interface_path = tmp_path / "interface.json"
    units = ["mm", "mrad", "mm", "mrad"]
    if dimension == 6:
        units.append("ns" if longitudinal == "time_s" else "mm")
    if dimension > 4:
        units.append("1")
    request = {
        "state_units": units,
        "fixed_controls": {"field_scale": 1.0},
        "default_controls": {"effective_rigidity_tm": 1.0},
        "control_bounds": {"field_scale": [-1.0, 2.0], "effective_rigidity_tm": [-2.0, 2.0]},
    }
    write_json(native_path, native)
    write_json(request_path, request)
    export_interface(native_path, request_path, interface_path)
    model = {
        "public_interface": {
            "native_contract_path": str(native_path),
            "interface_path": str(interface_path),
        }
    }
    return transport, generation, protocol, model, selected


@pytest.mark.parametrize(
    "dimension,longitudinal",
    [(4, "path_m"), (5, "path_m"), (6, "path_m"), (6, "ct_m"), (6, "time_s")],
)
def test_public_units_convert_ranges_and_fixed_controls_are_not_variable_inputs(
    tmp_path, dimension, longitudinal
):
    transport, generation, protocol, model, selected = public_case(
        tmp_path, dimension, longitudinal
    )
    report = collect(
        transport, generation, protocol, model, protocol_sha256="a" * 64, selected=selected
    )
    assert report["interface"] == "public"
    assert report["beam_parameters"][0]["rough_valid_range"] == pytest.approx([-20, 30])
    assert report["beam_parameters"][1]["rough_valid_range"] == pytest.approx([-10, 10])
    assert report["beam_parameters"][1]["unit"] == "mrad"
    assert report["native_beam_parameters"][1]["unit"] == "rad"
    assert report["model_input_dimensions"] == dimension + 1
    assert report["fixed_controls"] == {"field_scale": 1.0}
    assert report["default_controls"] == {"effective_rigidity_tm": 1.0}
    assert len(report["magnet_parameters"]) == 1
    control = report["magnet_parameters"][0]
    assert control["rough_valid_range"] == [-1.2, 1.1]
    assert control["api_bounds"] == [-2.0, 2.0]
    if dimension == 6 and longitudinal == "time_s":
        assert report["beam_parameters"][4]["unit"] == "ns"
        assert report["beam_parameters"][4]["rough_valid_range"] == pytest.approx([-1, 1])
    assert "Underlying native model inputs" in render(report)


@pytest.mark.parametrize("change", ["protocol", "selected", "contract_bytes"])
def test_unbound_public_interface_fails(tmp_path, change):
    transport, generation, protocol, model, selected = public_case(tmp_path)
    if change == "contract_bytes":
        path = model["public_interface"]["native_contract_path"]
        value = read_json(path)
        value["unexpected"] = "different contract bytes"
        write_json(path, value)
    expected = deepcopy(selected)
    if change == "selected":
        expected["seed"] = 9
    with pytest.raises(ValueError, match="not bound"):
        collect(
            transport,
            generation,
            protocol,
            model,
            protocol_sha256="b" * 64 if change == "protocol" else "a" * 64,
            selected=expected,
        )


def test_conflicting_frontmatter_cannot_relabel_units_or_input_count():
    transport, generation, protocol = case()
    with pytest.raises(ValueError, match="conflicts"):
        collect(transport, generation, protocol, {"frontmatter": {"model_input_dimensions": 5}})


def test_omitted_longitudinal_dimension_does_not_get_energy_or_time():
    transport, generation, protocol = case(4)
    text = render(collect(transport, generation, protocol, {}))
    assert "| delta_p |" not in text and "| time_s |" not in text


def test_extra_user_conditions_are_preserved_with_attribution():
    transport, generation, protocol = case()
    report = collect(
        transport,
        generation,
        protocol,
        {"frontmatter": {"input_constraints": ["Use the supplied pipe limits."]}},
    )
    assert (
        "Caller-supplied extra condition: Use the supplied pipe limits."
        in report["input_constraints"]
    )
