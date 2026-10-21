from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SOTAMethod:
    optimizer: str
    method: str
    source_basis: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _sota(optimizer: str, method: str, source_basis: str) -> SOTAMethod:
    return SOTAMethod(optimizer, method, source_basis)


CMA = _sota(
    "cma_es",
    "CMA-ES",
    "State-of-the-art covariance adaptation for continuous non-separable black-box problems.",
)
NGOPT = _sota(
    "ngopt",
    "Nevergrad NGOpt portfolio",
    "Nevergrad's plug-and-play adaptive optimizer API for continuous black-box problems.",
)
MIXED = _sota(
    "ng_mixed_portfolio",
    "Nevergrad PortfolioDiscreteOnePlusOne",
    "Nevergrad's source guidance for mixed and discrete parameter spaces.",
)
SMAC = _sota(
    "smac_hpo",
    "SMAC HyperparameterOptimizationFacade",
    "State-of-the-art ConfigSpace-native Bayesian optimization for hierarchical HPO.",
)
TBPSA = _sota(
    "tbpsa",
    "Nevergrad TBPSA",
    "Nevergrad's source-recommended optimizer for noisy objectives.",
)
ISRES = _sota(
    "pymoo_isres",
    "pymoo ISRES",
    "pymoo's plug-and-play improved stochastic-ranking optimizer for nonlinear constraints.",
)
BLEAQ = _sota(
    "source_bilevel",
    "Source-structured bilevel optimization",
    "The SMD source supplies analytical follower mappings; BOLIB quadratic followers use a constrained local solve, with CMA-ES at the leader level.",
)


# Exactly one SOTA method is declared for every benchmark task.
SOTA_METHODS: dict[str, SOTAMethod] = {
    "D01": CMA, "D02": CMA, "D03": CMA, "D04": NGOPT, "D05": NGOPT,
    "V01": MIXED, "V02": MIXED, "V03": MIXED, "V04": SMAC, "V05": SMAC,
    "N01": TBPSA, "N02": TBPSA, "N03": TBPSA, "N04": TBPSA, "N05": TBPSA,
    "S01": CMA, "S02": CMA, "S03": NGOPT, "S04": NGOPT, "S05": NGOPT,
    "C01": ISRES, "C02": ISRES, "C03": ISRES, "C04": ISRES, "C05": ISRES,
    "B01": BLEAQ, "B02": BLEAQ, "B03": BLEAQ, "B04": BLEAQ, "B05": BLEAQ,
}


def sota_method(problem_id: str) -> SOTAMethod:
    try:
        return SOTA_METHODS[problem_id.upper()]
    except KeyError as exc:
        raise KeyError(f"No SOTA method is registered for {problem_id!r}.") from exc


def list_sota_methods() -> list[dict[str, str]]:
    return [
        {"problem_id": problem_id, **method.to_dict()}
        for problem_id, method in SOTA_METHODS.items()
    ]
