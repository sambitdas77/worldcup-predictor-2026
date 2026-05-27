from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


MODEL_FEATURES = [
    "power_score",
    "attack_score",
    "creation_score",
    "defense_score",
    "keeper_score",
    "depth_score",
    "data_confidence_score",
]


def load_team_features(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_DIR / "final_team_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing team features: {path}")
    teams = pd.read_csv(path, encoding="latin-1")
    print(f"Loaded team features: {teams.shape[0]} teams x {teams.shape[1]} columns")
    print(list(teams.columns))
    return teams


def sigmoid(value: float | pd.Series) -> float | pd.Series:
    return 1 / (1 + np.exp(-value))


def expected_goal_strength(row: pd.Series) -> float:
    return (
        0.45 * row.get("attack_score", 50)
        + 0.25 * row.get("creation_score", 50)
        + 0.15 * row.get("depth_score", 50)
        + 0.15 * row.get("power_score", 50)
    )


def defensive_resistance(row: pd.Series) -> float:
    return (
        0.45 * row.get("defense_score", 50)
        + 0.30 * row.get("keeper_score", 50)
        + 0.15 * row.get("depth_score", 50)
        + 0.10 * row.get("power_score", 50)
    )


def predict_match(team_a: pd.Series, team_b: pd.Series, draw_scale: float = 0.24) -> dict[str, float | str]:
    """Baseline probability model.

    This is not trained yet. It converts team feature differences into
    probabilities, which gives us a reasonable first model before we add
    historical match results.
    """

    attack_edge_a = expected_goal_strength(team_a) - defensive_resistance(team_b)
    attack_edge_b = expected_goal_strength(team_b) - defensive_resistance(team_a)
    power_edge = team_a.get("power_score", 50) - team_b.get("power_score", 50)
    net_edge = 0.55 * power_edge + 0.45 * (attack_edge_a - attack_edge_b)

    non_draw_a = float(sigmoid(net_edge / 12.0))
    closeness = abs(net_edge)
    draw_prob = float(0.30 * np.exp(-closeness / 22.0) + draw_scale * 0.25)
    draw_prob = float(np.clip(draw_prob, 0.08, 0.34))

    remaining = 1 - draw_prob
    team_a_win = remaining * non_draw_a
    team_b_win = remaining * (1 - non_draw_a)

    return {
        "team_a": team_a["nation_name"],
        "team_b": team_b["nation_name"],
        "team_a_code": team_a["nation_code"],
        "team_b_code": team_b["nation_code"],
        "team_a_win_prob": team_a_win,
        "draw_prob": draw_prob,
        "team_b_win_prob": team_b_win,
        "net_edge": net_edge,
    }


def build_pairwise_predictions(teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clean = teams.dropna(subset=["nation_code", "nation_name"]).reset_index(drop=True)
    for i, j in combinations(range(len(clean)), 2):
        rows.append(predict_match(clean.loc[i], clean.loc[j]))
    return pd.DataFrame(rows).sort_values("net_edge", ascending=False)


def contender_table(teams: pd.DataFrame, top_n: int = 16) -> pd.DataFrame:
    cols = [
        "nation_code",
        "nation_name",
        "power_score",
        "attack_score",
        "creation_score",
        "defense_score",
        "keeper_score",
        "depth_score",
        "data_confidence_score",
        "selection_method",
    ]
    cols = [col for col in cols if col in teams.columns]
    return teams[cols].sort_values("power_score", ascending=False).head(top_n)


def read_historical_results(path: Path | None = None) -> pd.DataFrame | None:
    path = path or RAW_DIR / "results.csv"
    if not path.exists():
        print(f"Historical results not found at {path}")
        return None

    results = pd.read_csv(path, encoding="latin-1")
    print(f"Loaded historical results: {results.shape[0]} rows x {results.shape[1]} columns")
    print(list(results.columns))
    return results


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    teams = load_team_features()
    pairwise = build_pairwise_predictions(teams)
    contenders = contender_table(teams)
    results = read_historical_results()

    pairwise_path = PROCESSED_DIR / "baseline_pairwise_match_predictions.csv"
    contenders_path = PROCESSED_DIR / "contender_power_ranking.csv"
    pairwise.to_csv(pairwise_path, index=False)
    contenders.to_csv(contenders_path, index=False)

    print(f"Saved pairwise predictions -> {pairwise_path}")
    print(f"Saved contender ranking -> {contenders_path}")
    print("\nTop contenders:")
    print(contenders.to_string(index=False))

    if results is None:
        print("\nNext data needed for trained ML: data/raw/results.csv")


if __name__ == "__main__":
    main()
