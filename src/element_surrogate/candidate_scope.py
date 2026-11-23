"""One native-study eligibility binding for generated and supplied labels."""


def validate_candidate_scope(config, description, plan=None):
    """Reuse measured numerical authorities, not the label origin as eligibility.

    Runtime checkpoint loading remains backwards compatible. A new study needs
    current implementation evidence and the declared transport. When a full
    source envelope is provided, bind that too; actual cohorts are additionally
    checked against candidate input bounds by the neural adapter.
    """
    dimension = config["dimension"]
    structured = (
        config.get("model_family") == "canonical_shear"
        or config["base"].startswith("theory_")
        or len(config.get("reflection_group", [])) > 1
    )
    if structured and description is None:
        raise ValueError(
            "structured native candidates require their declared transport description"
        )
    scope = None
    if plan is not None:
        if plan.get("transport") != description:
            raise ValueError("source envelope differs from the declared transport")
        scope = {
            "transport": description,
            "ranges": plan["ranges"],
            "settings": [
                {k: s[k] for k in ("id", "effective_rigidity_tm", "field_scale")}
                for s in plan["settings"]
            ],
            "effective_coupling_range": plan.get("effective_coupling_range"),
        }
    if config.get("model_family") == "canonical_shear":
        from element_surrogate.canonical import validate

        validate(config, expected_scope=scope, current=True)
        if config["canonical_contract"]["scope"]["transport"] != description:
            raise ValueError("canonical assessment belongs to another transport")
    if config["base"].startswith("theory_"):
        from element_surrogate.theory import validate

        kind = "drift" if config["base"] == "theory_drift" else "paraxial_quadrupole"
        validate(
            config["theory_contract"],
            kind,
            dimension,
            expected_transport=description,
            current=True,
            expected_scope=scope,
        )
    if len(config.get("reflection_group", [])) > 1:
        from element_surrogate.symmetry import validate, validate_measurement

        validator = (
            validate_measurement
            if config.get("model_family") == "reflection_features"
            and config.get("symmetry_breaking")
            else validate
        )
        evidence = config["symmetry_assessment"]
        validator(
            evidence,
            config["reflection_group"],
            dimension,
            expected_scope=scope,
            require_current=True,
        )
        if evidence["scope"]["transport"] != description:
            raise ValueError("symmetry assessment belongs to another transport")
