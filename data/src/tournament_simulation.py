from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from trained_match_model import (
    NAME_TO_CODE,
    add_missing_team_placeholders,
    build_training_data,
    latest_elo_ratings,
    load_inputs,
    matchup_features,
    prepare_results,
    team_lookup,
    train_models,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANUAL_DIR = PROJECT_ROOT / "data" / "manual"
ANNEX_C_PATH = MANUAL_DIR / "fifa_annex_c_third_place_allocations.csv"


ROUND_OF_32 = [
    ("M73", "2A", "2B"),
    ("M76", "1C", "2F"),
    ("M74", "1E", "3ABCDF"),
    ("M75", "1F", "2C"),
    ("M78", "2E", "2I"),
    ("M77", "1I", "3CDFGH"),
    ("M79", "1A", "3CEFHI"),
    ("M80", "1L", "3EHIJK"),
    ("M82", "1G", "3AEHIJ"),
    ("M81", "1D", "3BEFIJ"),
    ("M84", "1H", "2J"),
    ("M83", "2K", "2L"),
    ("M85", "1B", "3EFGIJ"),
    ("M88", "2D", "2G"),
    ("M86", "1J", "2H"),
    ("M87", "1K", "3DEIJL"),
]

# Same column order used in Annexe C of the FIFA World Cup 26 regulations:
# 1A, 1B, 1D, 1E, 1G, 1I, 1K, 1L.
THIRD_PLACE_SLOT_ORDER = ["M79", "M85", "M81", "M74", "M82", "M77", "M87", "M80"]
ANNEX_C_SLOT_COLUMNS = {
    "M79": "M79_1A",
    "M85": "M85_1B",
    "M81": "M81_1D",
    "M74": "M74_1E",
    "M82": "M82_1G",
    "M77": "M77_1I",
    "M87": "M87_1K",
    "M80": "M80_1L",
}

ROUND_OF_16 = [
    ("M90", "M73", "M75"),
    ("M89", "M74", "M77"),
    ("M91", "M76", "M78"),
    ("M92", "M79", "M80"),
    ("M93", "M83", "M84"),
    ("M94", "M81", "M82"),
    ("M95", "M86", "M88"),
    ("M96", "M85", "M87"),
]

QUARTERS = [
    ("M97", "M89", "M90"),
    ("M98", "M93", "M94"),
    ("M99", "M91", "M92"),
    ("M100", "M95", "M96"),
]

SEMIS = [
    ("M101", "M97", "M98"),
    ("M102", "M99", "M100"),
]

FINAL = ("M104", "M101", "M102")


def model_classes(model) -> list[str]:
    return list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["model"].classes_)


def predict_match_probs(
    team_a_code: str,
    team_b_code: str,
    teams_by_code: dict[str, pd.Series],
    model,
    elo_ratings: dict[str, float],
) -> dict[str, float]:
    team_a = teams_by_code[team_a_code]
    team_b = teams_by_code[team_b_code]
    team_a_elo = elo_ratings.get(team_a_code, 1500.0)
    team_b_elo = elo_ratings.get(team_b_code, 1500.0)
    context = {
        "elo_diff": team_a_elo - team_b_elo,
        "elo_absdiff": abs(team_a_elo - team_b_elo),
        "home_advantage": 0,
        "is_neutral": 1,
        "is_world_cup": 1,
        "is_qualifier": 0,
        "is_friendly": 0,
        "is_continental": 0,
    }
    X = pd.DataFrame([matchup_features(team_a, team_b, context)]).fillna(0)
    return dict(zip(model_classes(model), model.predict_proba(X)[0]))


def build_probability_cache(
    team_codes: list[str],
    teams_by_code: dict[str, pd.Series],
    model,
    elo_ratings: dict[str, float],
) -> dict[tuple[str, str], dict[str, float]]:
    cache: dict[tuple[str, str], dict[str, float]] = {}
    for team_a in team_codes:
        for team_b in team_codes:
            if team_a == team_b:
                continue
            cache[(team_a, team_b)] = predict_match_probs(team_a, team_b, teams_by_code, model, elo_ratings)
    return cache


def sample_group_result(probs: dict[str, float], rng: np.random.Generator) -> str:
    labels = ["home_win", "draw", "away_win"]
    weights = np.array([probs.get(label, 0.0) for label in labels], dtype="float64")
    weights = weights / weights.sum() if weights.sum() else np.array([0.4, 0.2, 0.4])
    return str(rng.choice(labels, p=weights))


def knockout_win_probability(probs: dict[str, float]) -> float:
    a = float(probs.get("home_win", 0.0))
    b = float(probs.get("away_win", 0.0))
    draw = float(probs.get("draw", 0.0))
    decisive = a + b
    if decisive <= 0:
        return 0.5
    # Draws in knockouts go to extra time/penalties. Split the draw mass in
    # proportion to non-draw strength instead of treating it as an impossible result.
    return a + draw * (a / decisive)


def sample_knockout_winner(
    team_a_code: str,
    team_b_code: str,
    teams_by_code: dict[str, pd.Series],
    model,
    elo_ratings: dict[str, float],
    rng: np.random.Generator,
) -> str:
    probs = predict_match_probs(team_a_code, team_b_code, teams_by_code, model, elo_ratings)
    team_a_prob = knockout_win_probability(probs)
    return team_a_code if rng.random() < team_a_prob else team_b_code


def group_fixture_rows(fixtures: pd.DataFrame) -> pd.DataFrame:
    out = fixtures.copy()
    out["team_a_code"] = out["team_a"].map(NAME_TO_CODE)
    out["team_b_code"] = out["team_b"].map(NAME_TO_CODE)
    return out.dropna(subset=["team_a_code", "team_b_code"]).copy()


def simulate_groups(
    fixtures: pd.DataFrame,
    teams_by_code: dict[str, pd.Series],
    probability_cache: dict[tuple[str, str], dict[str, float]],
    rng: np.random.Generator,
) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    table: dict[tuple[str, str], dict[str, float]] = {}
    for _, fixture in group_fixture_rows(fixtures).iterrows():
        group = str(fixture["group"])
        a = str(fixture["team_a_code"])
        b = str(fixture["team_b_code"])
        for code in [a, b]:
            table.setdefault(
                (group, code),
                {"group": group, "team_code": code, "points": 0, "gd": 0, "gf": 0, "power": 0.0},
            )
            table[(group, code)]["power"] = float(teams_by_code[code].get("power_score", 0.0))

        probs = probability_cache[(a, b)]
        result = sample_group_result(probs, rng)
        if result == "home_win":
            table[(group, a)]["points"] += 3
            table[(group, a)]["gd"] += 1
            table[(group, b)]["gd"] -= 1
            table[(group, a)]["gf"] += 2
            table[(group, b)]["gf"] += 1
        elif result == "away_win":
            table[(group, b)]["points"] += 3
            table[(group, b)]["gd"] += 1
            table[(group, a)]["gd"] -= 1
            table[(group, b)]["gf"] += 2
            table[(group, a)]["gf"] += 1
        else:
            table[(group, a)]["points"] += 1
            table[(group, b)]["points"] += 1
            table[(group, a)]["gf"] += 1
            table[(group, b)]["gf"] += 1

    ranked_groups: dict[str, list[str]] = {}
    third_rows = []
    for group in sorted({key[0] for key in table}):
        rows = [row for (row_group, _), row in table.items() if row_group == group]
        rows = sorted(
            rows,
            key=lambda row: (
                row["points"],
                row["gd"],
                row["gf"],
                row["power"],
                rng.random(),
            ),
            reverse=True,
        )
        ranked_groups[group] = [str(row["team_code"]) for row in rows]
        third = rows[2].copy()
        third["third_rank_score"] = (third["points"], third["gd"], third["gf"], third["power"], rng.random())
        third_rows.append(third)

    third_rows = sorted(
        third_rows,
        key=lambda row: row["third_rank_score"],
        reverse=True,
    )
    return ranked_groups, third_rows[:8]


@lru_cache(maxsize=1)
def load_annex_c_allocations() -> pd.DataFrame:
    if not ANNEX_C_PATH.exists():
        raise FileNotFoundError(
            f"Missing FIFA Annex C allocation table: {ANNEX_C_PATH}. "
            "Regenerate it from the official regulations PDF before running the strict simulator."
        )

    annex = pd.read_csv(ANNEX_C_PATH, dtype=str)
    required = {"option", "qualified_groups", *ANNEX_C_SLOT_COLUMNS.values()}
    missing = sorted(required - set(annex.columns))
    if missing:
        raise ValueError(f"Annex C table is missing required columns: {missing}")
    if len(annex) != 495:
        raise ValueError(f"Annex C table should contain 495 rows, found {len(annex)}")
    if annex["qualified_groups"].duplicated().any():
        duplicates = annex.loc[annex["qualified_groups"].duplicated(), "qualified_groups"].tolist()
        raise ValueError(f"Annex C table has duplicate qualified-group keys: {duplicates[:5]}")
    return annex.set_index("qualified_groups", drop=False)


def allocate_third_place_slots(third_place_pool: list[dict[str, object]]) -> dict[str, str]:
    """Assign qualified third-place teams using FIFA's exact Annexe C table."""

    group_to_team = {str(team["group"]): str(team["team_code"]) for team in third_place_pool}
    qualified_groups = "".join(sorted(group_to_team))
    annex = load_annex_c_allocations()
    if qualified_groups not in annex.index:
        raise RuntimeError(f"No FIFA Annex C allocation found for third-place groups: {qualified_groups}")

    row = annex.loc[qualified_groups]
    allocation = {}
    for match_id, column in ANNEX_C_SLOT_COLUMNS.items():
        group = str(row[column]).replace("3", "")
        if group not in group_to_team:
            raise RuntimeError(
                f"Annex C option {row['option']} maps {match_id} to 3{group}, "
                f"but group {group} is not in the qualified third-place pool {qualified_groups}"
            )
        allocation[match_id] = group_to_team[group]
    return allocation


def resolve_slot(
    slot: str,
    ranked_groups: dict[str, list[str]],
    third_place_allocation: dict[str, str],
    match_id: str,
) -> str:
    if slot[0] in {"1", "2"}:
        rank = int(slot[0]) - 1
        group = slot[1]
        return ranked_groups[group][rank]

    if match_id not in third_place_allocation:
        raise RuntimeError(f"Missing allocated third-place team for {match_id} ({slot})")
    return third_place_allocation[match_id]


def simulate_bracket(
    ranked_groups: dict[str, list[str]],
    third_place_pool: list[dict[str, object]],
    teams_by_code: dict[str, pd.Series],
    probability_cache: dict[tuple[str, str], dict[str, float]],
    rng: np.random.Generator,
) -> tuple[str, dict[str, set[str]]]:
    winners: dict[str, str] = {}
    stages: dict[str, set[str]] = defaultdict(set)
    third_place_allocation = allocate_third_place_slots(third_place_pool)

    for match_id, slot_a, slot_b in ROUND_OF_32:
        a = resolve_slot(slot_a, ranked_groups, third_place_allocation, match_id)
        b = resolve_slot(slot_b, ranked_groups, third_place_allocation, match_id)
        stages["round_of_32"].update([a, b])
        probs = probability_cache[(a, b)]
        team_a_prob = knockout_win_probability(probs)
        winners[match_id] = a if rng.random() < team_a_prob else b

    for stage_name, round_spec in [
        ("round_of_16", ROUND_OF_16),
        ("quarterfinal", QUARTERS),
        ("semifinal", SEMIS),
    ]:
        for match_id, left_id, right_id in round_spec:
            a = winners[left_id]
            b = winners[right_id]
            stages[stage_name].update([a, b])
            probs = probability_cache[(a, b)]
            team_a_prob = knockout_win_probability(probs)
            winners[match_id] = a if rng.random() < team_a_prob else b

    _, left_id, right_id = FINAL
    a = winners[left_id]
    b = winners[right_id]
    stages["final"].update([a, b])
    probs = probability_cache[(a, b)]
    team_a_prob = knockout_win_probability(probs)
    champion = a if rng.random() < team_a_prob else b
    stages["champion"].add(champion)
    return champion, stages


def simulate_tournament(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    model,
    elo_ratings: dict[str, float],
    simulations: int = 10000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    teams_by_code = team_lookup(teams)
    all_codes = sorted(set(group_fixture_rows(fixtures)["team_a_code"]) | set(group_fixture_rows(fixtures)["team_b_code"]))
    probability_cache = build_probability_cache(all_codes, teams_by_code, model, elo_ratings)
    stage_counts = {stage: Counter() for stage in ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final", "champion"]}
    champion_counts = Counter()

    for _ in range(simulations):
        ranked_groups, third_place_pool = simulate_groups(fixtures, teams_by_code, probability_cache, rng)
        champion, stages = simulate_bracket(ranked_groups, third_place_pool, teams_by_code, probability_cache, rng)
        champion_counts[champion] += 1
        for stage, teams_in_stage in stages.items():
            for code in teams_in_stage:
                stage_counts[stage][code] += 1

    name_by_code = {str(row["nation_code"]): str(row["nation_name"]) for _, row in teams.iterrows()}
    rows = []
    for code in all_codes:
        rows.append(
            {
                "nation_code": code,
                "nation_name": name_by_code.get(code, code),
                "round_of_32_prob": stage_counts["round_of_32"][code] / simulations,
                "round_of_16_prob": stage_counts["round_of_16"][code] / simulations,
                "quarterfinal_prob": stage_counts["quarterfinal"][code] / simulations,
                "semifinal_prob": stage_counts["semifinal"][code] / simulations,
                "final_prob": stage_counts["final"][code] / simulations,
                "title_prob": champion_counts[code] / simulations,
                "titles": champion_counts[code],
            }
        )

    summary = pd.DataFrame(rows).sort_values("title_prob", ascending=False).reset_index(drop=True)
    stage_rows = []
    for stage, counts in stage_counts.items():
        for code, count in counts.items():
            stage_rows.append(
                {
                    "stage": stage,
                    "nation_code": code,
                    "nation_name": name_by_code.get(code, code),
                    "count": count,
                    "probability": count / simulations,
                }
            )
    stage_detail = pd.DataFrame(stage_rows).sort_values(["stage", "probability"], ascending=[True, False])
    return summary, stage_detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WC 2026 Monte Carlo tournament simulation.")
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    simulations = args.simulations
    teams, results, fixtures = load_inputs()
    fixture_codes = set(fixtures["team_a"].map(NAME_TO_CODE).dropna()) | set(fixtures["team_b"].map(NAME_TO_CODE).dropna())
    teams = add_missing_team_placeholders(teams, fixture_codes)
    recent_results = prepare_results(results, min_year=2018)
    elo_ratings = latest_elo_ratings(results)
    X, y = build_training_data(recent_results, teams)
    logistic, forest, metrics = train_models(X, y)
    best_model_name = metrics.sort_values("accuracy", ascending=False).iloc[0]["model"]
    best_model = logistic if best_model_name == "logistic_regression" else forest
    print(f"\nUsing best validation model for tournament simulation: {best_model_name}")

    summary, stage_detail = simulate_tournament(
        fixtures=fixtures,
        teams=teams,
        model=best_model,
        elo_ratings=elo_ratings,
        simulations=simulations,
        seed=args.seed,
    )

    summary_path = PROCESSED_DIR / "wc_2026_monte_carlo_title_probabilities.csv"
    stage_path = PROCESSED_DIR / "wc_2026_monte_carlo_stage_probabilities.csv"
    summary.to_csv(summary_path, index=False)
    stage_detail.to_csv(stage_path, index=False)

    print(f"\nMonte Carlo simulations: {simulations}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {stage_path}")
    print("\nTitle probabilities:")
    display_cols = [
        "nation_name",
        "title_prob",
        "final_prob",
        "semifinal_prob",
        "quarterfinal_prob",
        "round_of_16_prob",
    ]
    print(summary[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
