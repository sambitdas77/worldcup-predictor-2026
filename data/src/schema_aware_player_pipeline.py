from __future__ import annotations

from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


WC_TEAMS = {
    "ENG": "England",
    "FRA": "France",
    "ESP": "Spain",
    "GER": "Germany",
    "POR": "Portugal",
    "NED": "Netherlands",
    "BEL": "Belgium",
    "CRO": "Croatia",
    "AUT": "Austria",
    "SUI": "Switzerland",
    "SWE": "Sweden",
    "NOR": "Norway",
    "SCO": "Scotland",
    "TUR": "Turkey",
    "CZE": "Czechia",
    "BIH": "Bosnia and Herzegovina",
    "ARG": "Argentina",
    "BRA": "Brazil",
    "URU": "Uruguay",
    "COL": "Colombia",
    "ECU": "Ecuador",
    "PAR": "Paraguay",
    "USA": "United States",
    "MEX": "Mexico",
    "CAN": "Canada",
    "PAN": "Panama",
    "CUW": "Curacao",
    "HAI": "Haiti",
    "MAR": "Morocco",
    "SEN": "Senegal",
    "GHA": "Ghana",
    "EGY": "Egypt",
    "TUN": "Tunisia",
    "CIV": "Cote d'Ivoire",
    "COD": "DR Congo",
    "CPV": "Cabo Verde",
    "DZA": "Algeria",
    "RSA": "South Africa",
    "JPN": "Japan",
    "KOR": "South Korea",
    "IRN": "Iran",
    "SAU": "Saudi Arabia",
    "AUS": "Australia",
    "QAT": "Qatar",
    "UZB": "Uzbekistan",
    "JOR": "Jordan",
    "IRQ": "Iraq",
    "NZL": "New Zealand",
}


ALIASES = {
    "player": ["Player", "player_name"],
    "nation": ["Nation"],
    "position": ["Pos", "position"],
    "squad": ["Squad", "team_title"],
    "competition": ["Comp", "league_name"],
    "minutes": ["Min", "time"],
    "nineties": ["90s"],
    "matches": ["MP", "games"],
    "goals": ["Gls", "goals"],
    "assists": ["Ast", "assists"],
    "stat_source": ["source"],
    "xg": ["xG", "xG_est"],
    "npxg": ["npxG", "npxG"],
    "xag": ["xAG"],
    "xa": ["xA"],
    "shots": ["Sh", "shots"],
    "shots_on_target": ["SoT"],
    "key_passes": ["key_passes"],
    "progressive_passes": ["PrgP"],
    "progressive_carries": ["PrgC"],
    "progressive_receptions": ["PrgR"],
    "interceptions": ["Int"],
    "tackles_won": ["TklW"],
    "save_pct": ["Save%"],
    "goals_against": ["GA"],
    "clean_sheets": ["CS"],
    "xg_chain": ["xGChain"],
    "xg_buildup": ["xGBuildup"],
    "season": ["season"],
    "season_year": ["season_year"],
}


def print_columns(df: pd.DataFrame, label: str) -> None:
    print(f"\n========== {label} columns ({len(df.columns)}) ==========")
    print(list(df.columns))


def read_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    print(f"{label}: {df.shape[0]} rows x {df.shape[1]} columns")
    print_columns(df, label)
    return df


def read_optional_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"{label}: not found at {path}")
        return None
    return read_csv(path, label)


def write_csv_safely(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"WARNING: {path.name} was locked, wrote {fallback.name} instead")
        return fallback


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def extract_nation_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    parts = str(value).strip().split()
    if not parts:
        return None
    return parts[-1].upper()


def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def canonicalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["dataset_source"] = label

    for canonical, candidates in ALIASES.items():
        col = first_existing(df, candidates)
        if col is not None:
            out[canonical] = df[col]

    if "player" in out.columns:
        out["player_key"] = out["player"].map(normalize_name)

    if "nation_code" in df.columns:
        out["nation_code"] = df["nation_code"].where(df["nation_code"].notna(), None)
        out["nation_code"] = out["nation_code"].map(
            lambda value: str(value).upper().strip() if value is not None else None
        )
        out.loc[out["nation_code"].isin(["", "NAN", "NONE"]), "nation_code"] = None
    if "nation" in out.columns:
        derived_nation = out["nation"].map(extract_nation_code)
        if "nation_code" in out.columns:
            out["nation_code"] = out["nation_code"].fillna(derived_nation)
        else:
            out["nation_code"] = derived_nation

    if "minutes" in out.columns and "nineties" not in out.columns:
        out["nineties"] = to_numeric(out["minutes"]) / 90.0

    numeric_cols = [
        "minutes",
        "nineties",
        "matches",
        "goals",
        "assists",
        "xg",
        "npxg",
        "xag",
        "xa",
        "shots",
        "shots_on_target",
        "key_passes",
        "progressive_passes",
        "progressive_carries",
        "progressive_receptions",
        "interceptions",
        "tackles_won",
        "save_pct",
        "goals_against",
        "clean_sheets",
        "xg_chain",
        "xg_buildup",
        "season_year",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = to_numeric(out[col])

    return out


def prefix_feature_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = {"player_key", "nation_code"}
    renamed = {}
    for col in df.columns:
        if col not in keep:
            renamed[col] = f"{prefix}_{col}"
    return df.rename(columns=renamed)


def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "nineties" not in out.columns:
        return out

    denom = to_numeric(out["nineties"]).clip(lower=0.1)
    for col in [
        "goals",
        "assists",
        "xg",
        "npxg",
        "xag",
        "xa",
        "shots",
        "shots_on_target",
        "key_passes",
        "progressive_passes",
        "progressive_carries",
        "interceptions",
        "tackles_won",
    ]:
        if col in out.columns:
            out[f"{col}_per90"] = to_numeric(out[col]) / denom
    return out


def aggregate_duplicate_players(df: pd.DataFrame) -> pd.DataFrame:
    if not {"player_key", "nation_code"}.issubset(df.columns):
        return df

    id_cols = ["player_key", "nation_code"]
    text_cols = [
        col
        for col in ["player", "position", "squad", "competition", "nation", "stat_source"]
        if col in df.columns
    ]
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"season_year"}
    ]

    agg = {col: "first" for col in text_cols}
    agg.update({col: "sum" for col in numeric_cols})
    if "season_year" in df.columns:
        agg["season_year"] = "max"

    return (
        df.dropna(subset=id_cols)
        .groupby(id_cols, as_index=False, dropna=False)
        .agg(agg)
    )


def build_historical_trends(hist: pd.DataFrame, player_nation_map: pd.DataFrame) -> pd.DataFrame:
    if "player_key" not in hist.columns:
        return pd.DataFrame(columns=["player_key", "nation_code"])

    mapped = hist.merge(player_nation_map, on="player_key", how="inner")
    if mapped.empty:
        return pd.DataFrame(columns=["player_key", "nation_code"])

    recent = mapped.copy()
    if "season_year" in recent.columns:
        max_year = recent["season_year"].max()
        recent = recent[recent["season_year"] >= max_year - 2]

    group_cols = ["player_key", "nation_code"]
    agg = {}
    for col in [
        "minutes",
        "matches",
        "goals",
        "assists",
        "xg",
        "xa",
        "shots",
        "key_passes",
        "xg_chain",
        "xg_buildup",
    ]:
        if col in recent.columns:
            agg[col] = "sum"

    trends = recent.groupby(group_cols, as_index=False).agg(agg)
    if "minutes" in trends.columns:
        denom = to_numeric(trends["minutes"]).div(90).clip(lower=0.1)
        for col in ["goals", "assists", "xg", "xa", "shots", "key_passes"]:
            if col in trends.columns:
                trends[f"hist_{col}_per90"] = to_numeric(trends[col]) / denom

    return prefix_feature_columns(trends, "hist")


def build_player_ml_dataframe(
    current_raw: pd.DataFrame,
    previous_raw: pd.DataFrame,
    historical_raw: pd.DataFrame,
    wc_teams: dict[str, str] = WC_TEAMS,
) -> pd.DataFrame:
    current = add_per90_features(canonicalize(current_raw, "current_2025_26"))
    previous = add_per90_features(canonicalize(previous_raw, "previous_2023_24"))
    historical = canonicalize(historical_raw, "historical_2014_2025")

    current = current[current.get("nation_code").isin(wc_teams.keys())].copy()
    previous = previous[previous.get("nation_code").isin(wc_teams.keys())].copy()

    current = aggregate_duplicate_players(current)
    previous = aggregate_duplicate_players(previous)

    player_nation_map = (
        pd.concat(
            [
                current[["player_key", "nation_code"]],
                previous[["player_key", "nation_code"]],
            ],
            ignore_index=True,
        )
        .dropna()
        .drop_duplicates()
    )

    historical_trends = build_historical_trends(historical, player_nation_map)

    current_prefixed = prefix_feature_columns(current, "cur")
    previous_prefixed = prefix_feature_columns(previous, "prev")

    merged = current_prefixed.merge(
        previous_prefixed,
        on=["player_key", "nation_code"],
        how="outer",
        validate="one_to_one",
    )
    merged = merged.merge(
        historical_trends,
        on=["player_key", "nation_code"],
        how="left",
        validate="one_to_one",
    )

    merged["nation_name"] = merged["nation_code"].map(wc_teams)
    merged["player_name"] = merged.get("cur_player").combine_first(merged.get("prev_player"))

    for col in merged.select_dtypes(include=[np.number]).columns:
        merged[col] = merged[col].fillna(0)

    recent_parts = []
    if "cur_goals_per90" in merged.columns:
        recent_parts.append(0.65 * merged["cur_goals_per90"])
    if "prev_goals_per90" in merged.columns:
        recent_parts.append(0.25 * merged["prev_goals_per90"])
    if "hist_hist_goals_per90" in merged.columns:
        recent_parts.append(0.10 * merged["hist_hist_goals_per90"])
    merged["weighted_goal_form"] = sum(recent_parts) if recent_parts else 0.0

    chance_parts = []
    for weight, col in [
        (0.40, "cur_xg_per90"),
        (0.20, "cur_xag_per90"),
        (0.20, "prev_xg_per90"),
        (0.10, "prev_xag_per90"),
        (0.10, "hist_hist_xg_per90"),
    ]:
        if col in merged.columns:
            chance_parts.append(weight * merged[col])
    merged["weighted_chance_form"] = sum(chance_parts) if chance_parts else 0.0

    minutes_parts = []
    if "cur_minutes" in merged.columns:
        minutes_parts.append(0.70 * merged["cur_minutes"])
    if "prev_minutes" in merged.columns:
        minutes_parts.append(0.30 * merged["prev_minutes"])
    merged["weighted_minutes"] = sum(minutes_parts) if minutes_parts else 0.0

    front_cols = [
        "player_key",
        "player_name",
        "nation_code",
        "nation_name",
        "weighted_goal_form",
        "weighted_chance_form",
        "weighted_minutes",
    ]
    front_cols = [col for col in front_cols if col in merged.columns]
    other_cols = [col for col in merged.columns if col not in front_cols]
    return merged[front_cols + other_cols].sort_values(
        ["nation_code", "weighted_chance_form", "weighted_goal_form"],
        ascending=[True, False, False],
    )


def append_supplemental_players(
    current_raw: pd.DataFrame,
    supplemental_raw: pd.DataFrame | None,
) -> pd.DataFrame:
    if supplemental_raw is None or supplemental_raw.empty:
        return current_raw

    current_keys = canonicalize(current_raw, "current_duplicate_check")
    supplemental_keys = canonicalize(supplemental_raw, "supplemental_duplicate_check")

    existing = set(
        current_keys[["player_key", "nation_code"]]
        .dropna()
        .itertuples(index=False, name=None)
    )
    supplemental_keys["_row_number"] = supplemental_keys.index
    duplicate_rows = supplemental_keys[
        supplemental_keys[["player_key", "nation_code"]]
        .apply(tuple, axis=1)
        .isin(existing)
    ]["_row_number"]

    filtered_supplemental = supplemental_raw.drop(index=duplicate_rows).copy()
    skipped = len(supplemental_raw) - len(filtered_supplemental)

    combined = pd.concat([current_raw, filtered_supplemental], ignore_index=True, sort=False)
    print(
        f"Added {len(filtered_supplemental)} supplemental players "
        f"({skipped} already existed in current data); "
        f"current season raw is now {len(combined)} rows"
    )
    return combined


def build_team_ml_dataframe(player_ml: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["nation_code", "nation_name"]
    numeric_cols = player_ml.select_dtypes(include=[np.number]).columns.tolist()

    agg = {col: "mean" for col in numeric_cols}
    for col in ["weighted_goal_form", "weighted_chance_form", "weighted_minutes"]:
        if col in numeric_cols:
            agg[col] = ["mean", "sum", "max"]

    team = player_ml.groupby(group_cols).agg(agg)
    team.columns = [
        "_".join(part for part in col if part).strip("_")
        for col in team.columns.to_flat_index()
    ]
    team = team.reset_index()

    if "player_key" in player_ml.columns:
        depth = player_ml.groupby(group_cols)["player_key"].count().reset_index(name="player_count")
        team = team.merge(depth, on=group_cols, how="left")

    all_teams = pd.DataFrame(
        [{"nation_code": code, "nation_name": name} for code, name in WC_TEAMS.items()]
    )
    team = all_teams.merge(team, on=group_cols, how="left")

    for col in team.select_dtypes(include=[np.number]).columns:
        team[col] = team[col].fillna(0)
    if "player_count" in team.columns:
        team["has_player_coverage"] = team["player_count"] > 0

    return team.sort_values("weighted_chance_form_sum", ascending=False)


def build_coverage_report(player_ml: pd.DataFrame) -> pd.DataFrame:
    grouped = player_ml.groupby(["nation_code", "nation_name"], as_index=False)
    agg_spec = {
        "player_count": ("player_key", "count"),
        "current_players": ("cur_player", lambda s: s.notna().sum()),
        "previous_players": ("prev_player", lambda s: s.notna().sum()),
        "historical_minutes": ("hist_minutes", "sum"),
    }
    if "cur_stat_source" in player_ml.columns:
        agg_spec["supplemental_players"] = (
            "cur_stat_source",
            lambda s: s.notna().sum(),
        )
        agg_spec["verified_supplemental_players"] = (
            "cur_stat_source",
            lambda s: (s == "manual_verified").sum(),
        )
        agg_spec["estimated_supplemental_players"] = (
            "cur_stat_source",
            lambda s: (s == "estimated_average").sum(),
        )

    coverage = grouped.agg(**agg_spec)
    all_teams = pd.DataFrame(
        [{"nation_code": code, "nation_name": name} for code, name in WC_TEAMS.items()]
    )
    coverage = all_teams.merge(coverage, on=["nation_code", "nation_name"], how="left")
    for col in [
        "player_count",
        "current_players",
        "previous_players",
        "historical_minutes",
        "supplemental_players",
        "verified_supplemental_players",
        "estimated_supplemental_players",
    ]:
        if col not in coverage.columns:
            coverage[col] = 0
        coverage[col] = coverage[col].fillna(0)
    coverage["coverage_level"] = np.select(
        [
            coverage["player_count"] == 0,
            coverage["player_count"] < 5,
            coverage["player_count"] < 15,
        ],
        ["missing", "very_low", "low"],
        default="usable",
    )
    return coverage.sort_values(["coverage_level", "player_count"])


def build_squad_player_coverage(
    squads_raw: pd.DataFrame | None,
    player_ml: pd.DataFrame,
) -> pd.DataFrame | None:
    if squads_raw is None or squads_raw.empty:
        return None

    required = {"nation_code", "player_name"}
    missing = required - set(squads_raw.columns)
    if missing:
        print(f"Squad coverage skipped; missing columns: {sorted(missing)}")
        return None

    squads = squads_raw.copy()
    squads["nation_code"] = squads["nation_code"].astype(str).str.upper().str.strip()
    squads["player_key"] = squads["player_name"].map(normalize_name)

    available = player_ml[
        [
            col
            for col in [
                "player_key",
                "nation_code",
                "weighted_goal_form",
                "weighted_chance_form",
                "weighted_minutes",
                "cur_stat_source",
            ]
            if col in player_ml.columns
        ]
    ].drop_duplicates(["player_key", "nation_code"])

    covered = squads.merge(available, on=["player_key", "nation_code"], how="left")
    covered["has_stats"] = covered["weighted_minutes"].notna()
    covered["is_supplemental"] = covered.get("cur_stat_source", pd.Series(index=covered.index)).notna()
    for col in ["weighted_goal_form", "weighted_chance_form", "weighted_minutes"]:
        if col in covered.columns:
            covered[col] = covered[col].fillna(0)
    return covered


def build_squad_nation_coverage(squad_player_coverage: pd.DataFrame | None) -> pd.DataFrame | None:
    if squad_player_coverage is None or squad_player_coverage.empty:
        return None

    report = (
        squad_player_coverage.groupby("nation_code", as_index=False)
        .agg(
            squad_players=("player_key", "count"),
            covered_players=("has_stats", "sum"),
            supplemental_players=("is_supplemental", "sum"),
        )
    )
    report["coverage_pct"] = (
        report["covered_players"] / report["squad_players"].replace(0, np.nan)
    ).fillna(0)
    report["nation_name"] = report["nation_code"].map(WC_TEAMS)
    cols = [
        "nation_code",
        "nation_name",
        "squad_players",
        "covered_players",
        "supplemental_players",
        "coverage_pct",
    ]
    return report[cols].sort_values(["coverage_pct", "covered_players"], ascending=[True, True])


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    current_raw = read_csv(RAW_DIR / "players_data-2025_2026.csv", "2025-26 current season")
    previous_raw = read_csv(RAW_DIR / "players_data-2023_2024.csv", "2023-24 previous season")
    historical_raw = read_csv(RAW_DIR / "players_data-2014_2025.csv", "2014-2025 historical trends")
    supplemental_raw = read_optional_csv(
        RAW_DIR / "supplemental_players_2025_26.csv",
        "2025-26 supplemental players",
    )
    squads_raw = read_optional_csv(
        RAW_DIR / "wc_squads_parsed.csv",
        "parsed World Cup squads",
    )
    current_raw = append_supplemental_players(current_raw, supplemental_raw)

    player_ml = build_player_ml_dataframe(current_raw, previous_raw, historical_raw)
    team_ml = build_team_ml_dataframe(player_ml)
    coverage = build_coverage_report(player_ml)
    squad_player_coverage = build_squad_player_coverage(squads_raw, player_ml)
    squad_nation_coverage = build_squad_nation_coverage(squad_player_coverage)

    player_path = PROCESSED_DIR / "player_ml_features.csv"
    team_path = PROCESSED_DIR / "team_ml_features.csv"
    coverage_path = PROCESSED_DIR / "coverage_report.csv"
    squad_player_path = PROCESSED_DIR / "squad_player_coverage.csv"
    squad_nation_path = PROCESSED_DIR / "squad_nation_coverage.csv"
    player_path = write_csv_safely(player_ml, player_path)
    team_path = write_csv_safely(team_ml, team_path)
    coverage_path = write_csv_safely(coverage, coverage_path)
    if squad_player_coverage is not None:
        squad_player_path = write_csv_safely(squad_player_coverage, squad_player_path)
    if squad_nation_coverage is not None:
        squad_nation_path = write_csv_safely(squad_nation_coverage, squad_nation_path)

    print("\n========== ML output ==========")
    print(f"Player ML dataframe: {player_ml.shape} -> {player_path}")
    print(f"Team ML dataframe: {team_ml.shape} -> {team_path}")
    print(f"Coverage report: {coverage.shape} -> {coverage_path}")
    if squad_player_coverage is not None:
        print(f"Squad player coverage: {squad_player_coverage.shape} -> {squad_player_path}")
    if squad_nation_coverage is not None:
        print(f"Squad nation coverage: {squad_nation_coverage.shape} -> {squad_nation_path}")
    print("\nLowest coverage teams:")
    print(coverage.head(12).to_string(index=False))
    if squad_nation_coverage is not None:
        print("\nLowest announced-squad coverage:")
        print(squad_nation_coverage.head(12).to_string(index=False))
    print("\nTop team rows:")
    print(team_ml.head(15).to_string())


if __name__ == "__main__":
    main()
