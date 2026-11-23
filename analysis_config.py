"""Shared configuration and output used by the notebook and the command line."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

SECTORS = ("Banks L1", "Technology L1", "Consumer Goods L1", "Real Estate L2", "Steel L4")


@dataclass(frozen=True)
class AnalysisConfig:
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    output_dir: Path = PROJECT_ROOT / "outputs_sector_volatility"
    targets: tuple[str, ...] = SECTORS
    test_fraction: float = 0.25
    max_window: int = 1000
    refit: int = 20
    significance: float = 0.05


class AnalysisOutput:
    """Save analysis outputs and optionally render them inline."""

    def __init__(self, output_dir: Path, *, render: Callable = print, inline: bool = False):
        self.directory = output_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        self.render = render
        self.inline = inline
        pd.set_option("display.precision", 4)
        pd.set_option("display.max_columns", 24)
        plt.rcParams.update({
            "figure.figsize": (11, 4),
            "figure.dpi": 95,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.20,
            "axes.labelcolor": "#24344a",
            "text.color": "#17283e",
            "axes.titleweight": "semibold",
        })

    def display(self, *objects):
        for value in objects:
            self.render(value)

    def save_table(self, data, filename: str, index: bool = True):
        """Keep full numeric precision and encode the heading for Excel."""
        data.to_csv(self.directory / filename, index=index, encoding="utf-8-sig", date_format="%Y-%m-%d")

    def show_figure(self, figure, filename: str):
        figure.savefig(self.directory / filename, dpi=175, bbox_inches="tight", facecolor="white")
        if self.inline:
            plt.show()
        plt.close(figure)
