"""Repository-root path resolution and input/output configuration.

All artifact paths follow model/IMPLEMENTATION_CONTRACT.md. The config is
resolved against a repo root (default: the directory the CLI is run from).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class Config:
    """Paths for all inputs and outputs of the forecasting pipeline."""

    root: str = "."
    data_dir_name: str = "data"
    model_dir_name: str = "model"

    # -- inputs -----------------------------------------------------------------
    external_features: str = "final_data/features/external_state_features.csv.gz"
    events: str = "final_data/events/events.csv.gz"
    entities: str = "final_data/entities/entities.csv.gz"
    relationships: str = "final_data/entities/relationships.csv.gz"
    drug_dictionary: str = "final_data/entities/drug_dictionary.json"
    rxnav_ndc_catalog: str = "final_data/entities/rxnav_ndc_catalog.csv.gz"

    partd_provider_drug: str = (
        "targeted_additions/cms_partd_prescriber_provider_drug/"
        "data/arkansas_partd_provider_drug_by_year.csv.gz"
    )
    fda_ndc_products: str = (
        "targeted_additions/fda_ndc_directory/data/fda_ndc_products_current.csv.gz"
    )
    fda_shortages: str = (
        "targeted_additions/fda_shortages_recalls_current/data/fda_shortages_current.csv.gz"
    )
    fda_enforcement: str = (
        "targeted_additions/fda_shortages_recalls_current/data/"
        "fda_enforcement_2023_current.csv.gz"
    )
    fda_establishments: str = "targeted_additions/fda_establishments/raw/drls_reg.zip"
    disease_surveillance_dir: str = (
        "targeted_additions/disease_surveillance_current/data"
    )
    news_dir: str = "targeted_additions/arkansas_news_3dlnews/data"
    # Optional bridge from the separately maintained FLAN-T5 news-only model.
    # It is materialized into model/artifacts/news/ before panel construction.
    news_only_features: str = (
        "existing_models/news_signal_model/data/derived/signals_monthly.csv"
    )
    nppes_provider_locations: str = (
        "targeted_additions/nppes_provider_locations/data/"
        "arkansas_nppes_provider_locations.csv.gz"
    )
    arcos_retail_summary: str = (
        "targeted_additions/dea_arcos_arkansas/data/"
        "arcos_arkansas_retail_summary.csv.gz"
    )

    # -- outputs ----------------------------------------------------------------
    artifacts_dir_name: str = "artifacts"

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        self.repo_root: Path = root
        self.data_dir: Path = root / self.data_dir_name
        self.model_dir: Path = root / self.model_dir_name
        self.artifacts_dir: Path = self.model_dir / self.artifacts_dir_name

    # -- resolved helpers -------------------------------------------------------
    def data_path(self, rel: str) -> Path:
        """Resolve a repository-relative path under ``data/``."""
        return self.data_dir / rel

    def artifact_path(self, rel: str) -> Path:
        """Resolve a repository-relative path under ``model/artifacts/``."""
        return self.artifacts_dir / rel

    @property
    def panel_dir(self) -> Path:
        """Directory for annual and quarterly supervised panels."""
        return self.artifact_path("panel")

    @property
    def trained_dir(self) -> Path:
        """Directory for serialized model specifications."""
        return self.artifact_path("trained")

    @property
    def forecasts_dir(self) -> Path:
        """Directory for forecast CSV outputs."""
        return self.artifact_path("forecasts")

    @property
    def evaluation_dir(self) -> Path:
        """Directory for evaluation metrics and fold reports."""
        return self.artifact_path("evaluation")

    @property
    def metadata_dir(self) -> Path:
        """Directory for source and artifact metadata."""
        return self.artifact_path("metadata")

    @property
    def geography_dir(self) -> Path:
        """Directory for geography crosswalk and weather registries."""
        return self.artifact_path("geography")

    def input_paths(self) -> Dict[str, Path]:
        """Every required input path for build-panel."""
        return {
            "external_features": self.data_path(self.external_features),
            "events": self.data_path(self.events),
            "entities": self.data_path(self.entities),
            "relationships": self.data_path(self.relationships),
            "drug_dictionary": self.data_path(self.drug_dictionary),
            "partd_provider_drug": self.data_path(self.partd_provider_drug),
            "fda_ndc_products": self.data_path(self.fda_ndc_products),
            "fda_shortages": self.data_path(self.fda_shortages),
            "fda_enforcement": self.data_path(self.fda_enforcement),
            "nppes_provider_locations": self.data_path(self.nppes_provider_locations),
            "arcos_retail_summary": self.data_path(self.arcos_retail_summary),
        }

    def ensure_dirs(self) -> None:
        """Create the standard artifact directories if they are absent."""
        for d in (
            self.panel_dir,
            self.trained_dir,
            self.forecasts_dir,
            self.evaluation_dir,
            self.metadata_dir,
            self.geography_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def resolve_root(candidate: str | None) -> Path:
    """Find repo root containing a data/ directory, starting from candidate."""
    p = Path(candidate or ".").expanduser().resolve()
    while True:
        if (p / "data").is_dir() and (p / "model").is_dir():
            return p
        if p.parent == p:
            raise FileNotFoundError(
                f"Could not locate repo root (data/ and model/) from {candidate!r}"
            )
        p = p.parent
