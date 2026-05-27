from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def latest_file(pattern: str, directory: Path = PROCESSED_DIR) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def read_processed_csv(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    df = pd.read_csv(path, encoding="latin-1")
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(list(df.columns))
    return df


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[column].fillna("").astype(str)


def percentile_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(50.0, index=values.index)
    return values.rank(pct=True) * 100


def bounded_score(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").fillna(0).clip(lower=0, upper=100)


def add_player_selection_score(player_ml: pd.DataFrame) -> pd.DataFrame:
    players = player_ml.copy()
    players["player_minutes_score"] = percentile_score(numeric(players, "weighted_minutes"))
    players["player_goal_score"] = percentile_score(numeric(players, "weighted_goal_form"))
    players["player_chance_score"] = percentile_score(numeric(players, "weighted_chance_form"))

    players["selection_score"] = (
        0.50 * players["player_minutes_score"]
        + 0.25 * players["player_goal_score"]
        + 0.25 * players["player_chance_score"]
    )
    return players


def combined_position(players: pd.DataFrame) -> pd.Series:
    cur = text(players, "cur_position")
    prev = text(players, "prev_position")
    return cur.where(cur.str.len() > 0, prev).str.upper()


def top_n_mean(df: pd.DataFrame, column: str, n: int) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").fillna(0).sort_values(ascending=False)
    return float(values.head(n).mean()) if len(values) else 0.0


def top_n_sum(df: pd.DataFrame, column: str, n: int) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").fillna(0).sort_values(ascending=False)
    return float(values.head(n).sum()) if len(values) else 0.0


def aggregate_probable_squads(player_ml: pd.DataFrame, squad_size: int = 26) -> pd.DataFrame:
    players = add_player_selection_score(player_ml)
    players["position_group"] = combined_position(players)

    rows = []
    for (nation_code, nation_name), group in players.groupby(["nation_code", "nation_name"], dropna=False):
        squad = group.sort_values("selection_score", ascending=False).head(squad_size).copy()
        attackers = squad[squad["position_group"].str.contains("FW", na=False)]
        midfielders = squad[squad["position_group"].str.contains("MF", na=False)]
        defenders = squad[squad["position_group"].str.contains("DF", na=False)]
        keepers = squad[squad["position_group"].str.contains("GK", na=False)]

        estimated_count = int((text(squad, "cur_stat_source") == "estimated_average").sum())
        verified_count = int((text(squad, "cur_stat_source") == "manual_verified").sum())

        rows.append(
            {
                "nation_code": nation_code,
                "nation_name": nation_name,
                "probable_squad_players": len(squad),
                "estimated_players_top26": estimated_count,
                "verified_supplemental_top26": verified_count,
                "avg_selection_score_top26": float(squad["selection_score"].mean()) if len(squad) else 0.0,
                "top6_goal_form_sum": top_n_sum(squad, "weighted_goal_form", 6),
                "top6_goal_form_mean": top_n_mean(squad, "weighted_goal_form", 6),
                "top10_chance_form_sum": top_n_sum(squad, "weighted_chance_form", 10),
                "top10_chance_form_mean": top_n_mean(squad, "weighted_chance_form", 10),
                "attacker_goal_form_mean": float(numeric(attackers, "weighted_goal_form").mean()) if len(attackers) else 0.0,
                "midfielder_chance_form_mean": float(numeric(midfielders, "weighted_chance_form").mean()) if len(midfielders) else 0.0,
                "defender_actions_per90_mean": float(
                    (
                        numeric(defenders, "cur_interceptions_per90")
                        + numeric(defenders, "cur_tackles_won_per90")
                    ).mean()
                )
                if len(defenders)
                else 0.0,
                "keeper_save_pct_best": float(numeric(keepers, "cur_save_pct").max()) if len(keepers) else 0.0,
                "squad_weighted_minutes_mean": float(numeric(squad, "weighted_minutes").mean()) if len(squad) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def add_player_keys_to_squads(squad_coverage: pd.DataFrame | None) -> pd.DataFrame | None:
    if squad_coverage is None or squad_coverage.empty:
        return None
    if not {"nation_code", "player_name"}.issubset(squad_coverage.columns):
        return None

    squads = squad_coverage.copy()
    squads["nation_code"] = squads["nation_code"].astype(str).str.upper().str.strip()
    squads["player_key"] = squads["player_name"].map(
        lambda value: "" if pd.isna(value) else " ".join(str(value).lower().split())
    )
    from schema_aware_player_pipeline import normalize_name

    squads["player_key"] = squads["player_name"].map(normalize_name)
    return squads[["nation_code", "player_key"]].dropna().drop_duplicates()


def aggregate_team_squads(
    player_ml: pd.DataFrame,
    squad_player_coverage: pd.DataFrame | None = None,
    squad_size: int = 26,
) -> pd.DataFrame:
    """Use official parsed squads when present, otherwise fall back to strongest top-N players."""

    players = add_player_selection_score(player_ml)
    players["position_group"] = combined_position(players)

    official_keys = add_player_keys_to_squads(squad_player_coverage)
    official_nations = set(official_keys["nation_code"]) if official_keys is not None else set()

    rows = []
    for (nation_code, nation_name), group in players.groupby(["nation_code", "nation_name"], dropna=False):
        if nation_code in official_nations:
            keys = official_keys[official_keys["nation_code"] == nation_code][["player_key", "nation_code"]]
            squad = keys.merge(group, on=["player_key", "nation_code"], how="inner")
            selection_method = "official_squad"

            if len(squad) == 0:
                squad = group.sort_values("selection_score", ascending=False).head(squad_size).copy()
                selection_method = "official_squad_unmatched_fallback_top26"
            elif len(squad) > squad_size:
                squad = squad.sort_values("selection_score", ascending=False).head(squad_size).copy()
                selection_method = "official_squad_capped_top26"
        else:
            squad = group.sort_values("selection_score", ascending=False).head(squad_size).copy()
            selection_method = "probable_top26"

        attackers = squad[squad["position_group"].str.contains("FW", na=False)]
        midfielders = squad[squad["position_group"].str.contains("MF", na=False)]
        defenders = squad[squad["position_group"].str.contains("DF", na=False)]
        keepers = squad[squad["position_group"].str.contains("GK", na=False)]

        estimated_count = int((text(squad, "cur_stat_source") == "estimated_average").sum())
        verified_count = int((text(squad, "cur_stat_source") == "manual_verified").sum())

        rows.append(
            {
                "nation_code": nation_code,
                "nation_name": nation_name,
                "selection_method": selection_method,
                "squad_players_used": len(squad),
                "estimated_players_used": estimated_count,
                "verified_supplemental_used": verified_count,
                "avg_selection_score": float(squad["selection_score"].mean()) if len(squad) else 0.0,
                "top6_goal_form_sum": top_n_sum(squad, "weighted_goal_form", 6),
                "top6_goal_form_mean": top_n_mean(squad, "weighted_goal_form", 6),
                "top10_chance_form_sum": top_n_sum(squad, "weighted_chance_form", 10),
                "top10_chance_form_mean": top_n_mean(squad, "weighted_chance_form", 10),
                "attacker_goal_form_mean": float(numeric(attackers, "weighted_goal_form").mean()) if len(attackers) else 0.0,
                "midfielder_chance_form_mean": float(numeric(midfielders, "weighted_chance_form").mean()) if len(midfielders) else 0.0,
                "defender_actions_per90_mean": float(
                    (
                        numeric(defenders, "cur_interceptions_per90")
                        + numeric(defenders, "cur_tackles_won_per90")
                    ).mean()
                )
                if len(defenders)
                else 0.0,
                "keeper_save_pct_best": float(numeric(keepers, "cur_save_pct").max()) if len(keepers) else 0.0,
                "squad_weighted_minutes_mean": float(numeric(squad, "weighted_minutes").mean()) if len(squad) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_power_scores(probable_squads: pd.DataFrame, squad_coverage: pd.DataFrame | None = None) -> pd.DataFrame:
    df = probable_squads.copy()
    selection_strength = numeric(df, "avg_selection_score")
    if "avg_selection_score_top26" in df.columns:
        selection_strength = numeric(df, "avg_selection_score_top26")

    df["attack_score"] = (
        0.40 * percentile_score(numeric(df, "top6_goal_form_sum"))
        + 0.25 * percentile_score(numeric(df, "attacker_goal_form_mean"))
        + 0.20 * percentile_score(numeric(df, "top10_chance_form_sum"))
        + 0.15 * percentile_score(selection_strength)
    )

    df["creation_score"] = (
        0.55 * percentile_score(numeric(df, "top10_chance_form_sum"))
        + 0.25 * percentile_score(numeric(df, "midfielder_chance_form_mean"))
        + 0.20 * percentile_score(numeric(df, "squad_weighted_minutes_mean"))
    )

    df["defense_score"] = (
        0.70 * percentile_score(numeric(df, "defender_actions_per90_mean"))
        + 0.30 * percentile_score(numeric(df, "squad_weighted_minutes_mean"))
    )

    df["keeper_score"] = percentile_score(numeric(df, "keeper_save_pct_best"))
    df["depth_score"] = percentile_score(selection_strength)

    df["data_confidence_score"] = 100.0
    estimated_count = numeric(df, "estimated_players_used")
    if "estimated_players_top26" in df.columns:
        estimated_count = numeric(df, "estimated_players_top26")

    squad_count = numeric(df, "squad_players_used")
    if "probable_squad_players" in df.columns:
        squad_count = numeric(df, "probable_squad_players")

    estimated_ratio = estimated_count / squad_count.replace(0, np.nan)
    estimated_ratio = estimated_ratio.fillna(1.0)
    df["data_confidence_score"] = df["data_confidence_score"] - (estimated_ratio * 35)

    if squad_coverage is not None and not squad_coverage.empty:
        coverage_cols = ["nation_code", "coverage_pct"]
        available = [col for col in coverage_cols if col in squad_coverage.columns]
        if set(coverage_cols).issubset(available):
            df = df.merge(squad_coverage[coverage_cols], on="nation_code", how="left")
            df["has_announced_squad_file"] = df["coverage_pct"].notna()
            df["coverage_pct"] = numeric(df, "coverage_pct", default=1.0)
            df["data_confidence_score"] = df["data_confidence_score"] * df["coverage_pct"].where(
                df["coverage_pct"] > 0,
                1.0,
            )
    else:
        df["coverage_pct"] = 1.0
        df["has_announced_squad_file"] = False

    df["data_confidence_score"] = bounded_score(df["data_confidence_score"])
    confidence_multiplier = 0.75 + (df["data_confidence_score"] / 100 * 0.25)

    df["raw_power_score"] = (
        0.35 * df["attack_score"]
        + 0.25 * df["creation_score"]
        + 0.18 * df["defense_score"]
        + 0.12 * df["keeper_score"]
        + 0.10 * df["depth_score"]
    )
    df["power_score"] = df["raw_power_score"] * confidence_multiplier

    return df.sort_values("power_score", ascending=False).reset_index(drop=True)


def main() -> None:
    player_ml = read_processed_csv("player_ml_features.csv")
    squad_coverage_path = PROCESSED_DIR / "squad_nation_coverage.csv"
    squad_coverage = pd.read_csv(squad_coverage_path) if squad_coverage_path.exists() else None
    squad_player_path = PROCESSED_DIR / "squad_player_coverage.csv"
    squad_player_coverage = pd.read_csv(squad_player_path) if squad_player_path.exists() else None

    team_squads = aggregate_team_squads(player_ml, squad_player_coverage)
    final_features = build_power_scores(team_squads, squad_coverage)

    probable_path = PROCESSED_DIR / "team_squad_features.csv"
    final_path = PROCESSED_DIR / "final_team_features.csv"
    team_squads.to_csv(probable_path, index=False)
    final_features.to_csv(final_path, index=False)

    print(f"team_squad_features: {team_squads.shape} -> {probable_path}")
    print(f"final_team_features: {final_features.shape} -> {final_path}")
    print(final_features.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
