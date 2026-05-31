from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANUAL_DIR = PROJECT_ROOT / "data" / "manual"


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


def read_manual_starting_xi_source(name: str = "manual_starting_xi.csv") -> pd.DataFrame | None:
    path = MANUAL_DIR / name
    if not path.exists():
        print(f"Manual XI file not found, using generated XI only: {path}")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"{name}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(list(df.columns))
    required = {"nation_code", "nation_name", "formation", "xi"}
    missing = required - set(df.columns)
    if missing:
        print(f"Manual XI skipped; missing columns: {sorted(missing)}")
        return None
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


def expand_manual_starting_xi(manual_xi_source: pd.DataFrame | None) -> pd.DataFrame | None:
    if manual_xi_source is None or manual_xi_source.empty:
        return None

    rows = []
    for _, team in manual_xi_source.iterrows():
        nation_code = str(team["nation_code"]).upper().strip()
        nation_name = str(team["nation_name"]).strip()
        formation = str(team["formation"]).strip()
        players = [part.strip() for part in str(team["xi"]).split(";") if part.strip()]

        for order, item in enumerate(players, start=1):
            if ":" in item:
                role, player_name = item.split(":", 1)
                role = role.strip().upper()
                player_name = player_name.strip()
            else:
                role = ""
                player_name = item.strip()

            rows.append(
                {
                    "nation_code": nation_code,
                    "nation_name": nation_name,
                    "formation": formation,
                    "xi_order": order,
                    "xi_role": role,
                    "player_name": player_name,
                }
            )

    expanded = pd.DataFrame(rows)
    if expanded.empty:
        return expanded

    from schema_aware_player_pipeline import normalize_name

    expanded["player_key"] = expanded["player_name"].map(normalize_name)
    return expanded


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


def select_team_squad_players(
    player_ml: pd.DataFrame,
    squad_player_coverage: pd.DataFrame | None = None,
    squad_size: int = 26,
) -> pd.DataFrame:
    players = add_player_selection_score(player_ml)
    players["position_group"] = combined_position(players)

    official_keys = add_player_keys_to_squads(squad_player_coverage)
    official_nations = set(official_keys["nation_code"]) if official_keys is not None else set()

    selected = []
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

        squad = squad.copy()
        squad["selection_method"] = selection_method
        squad["squad_rank"] = squad["selection_score"].rank(method="first", ascending=False).astype(int)
        selected.append(squad)

    if not selected:
        return pd.DataFrame()

    keep_cols = [
        "nation_code",
        "nation_name",
        "selection_method",
        "squad_rank",
        "player_name",
        "player_key",
        "position_group",
        "cur_squad",
        "prev_squad",
        "selection_score",
        "weighted_goal_form",
        "weighted_chance_form",
        "weighted_minutes",
        "cur_stat_source",
        "cur_save_pct",
        "cur_goals_per90",
        "cur_assists_per90",
        "cur_xg_per90",
        "cur_interceptions_per90",
        "cur_tackles_won_per90",
    ]
    result = pd.concat(selected, ignore_index=True, sort=False)
    keep_cols = [col for col in keep_cols if col in result.columns]
    return result[keep_cols].sort_values(["nation_code", "squad_rank"])


def choose_top_available(pool: pd.DataFrame, count: int, used_keys: set[str]) -> pd.DataFrame:
    candidates = pool[~pool["player_key"].isin(used_keys)].copy()
    if candidates.empty or count <= 0:
        return candidates.head(0)
    return candidates.sort_values("selection_score", ascending=False).head(count)


def build_probable_starting_xi(selected_squad_players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (nation_code, nation_name), squad in selected_squad_players.groupby(["nation_code", "nation_name"], dropna=False):
        squad = squad.copy()
        squad["position_group"] = squad["position_group"].fillna("").astype(str).str.upper()
        used: set[str] = set()

        gk_pool = squad[squad["position_group"].str.contains("GK", na=False)]
        df_pool = squad[squad["position_group"].str.contains("DF", na=False)]
        mf_pool = squad[squad["position_group"].str.contains("MF", na=False)]
        fw_pool = squad[squad["position_group"].str.contains("FW", na=False)]

        keeper_sort = gk_pool.assign(
            keeper_priority=numeric(gk_pool, "cur_save_pct") + numeric(gk_pool, "selection_score") / 100
        )
        chosen_groups = [
            ("GK", keeper_sort.sort_values("keeper_priority", ascending=False).head(1)),
            ("DF", choose_top_available(df_pool, 4, used)),
        ]

        for role, chosen in chosen_groups:
            for _, player in chosen.iterrows():
                used.add(player["player_key"])
                rows.append({**player.to_dict(), "xi_role": role})

        for role, pool, count in [("MF", mf_pool, 3), ("FW", fw_pool, 3)]:
            chosen = choose_top_available(pool, count, used)
            for _, player in chosen.iterrows():
                used.add(player["player_key"])
                rows.append({**player.to_dict(), "xi_role": role})

        current_count = sum(1 for row in rows if row["nation_code"] == nation_code)
        if current_count < 11:
            fallback = choose_top_available(squad, 11 - current_count, used)
            for _, player in fallback.iterrows():
                used.add(player["player_key"])
                rows.append({**player.to_dict(), "xi_role": "FLEX"})

    if not rows:
        return pd.DataFrame()

    xi = pd.DataFrame(rows)
    xi["xi_rank"] = xi.groupby("nation_code")["selection_score"].rank(method="first", ascending=False).astype(int)
    keep_cols = [
        "nation_code",
        "nation_name",
        "selection_method",
        "xi_role",
        "xi_rank",
        "player_name",
        "player_key",
        "position_group",
        "cur_squad",
        "prev_squad",
        "selection_score",
        "weighted_goal_form",
        "weighted_chance_form",
        "weighted_minutes",
        "cur_stat_source",
    ]
    keep_cols = [col for col in keep_cols if col in xi.columns]
    return xi[keep_cols].sort_values(["nation_code", "xi_role", "xi_rank"])


def build_manual_starting_xi(
    manual_xi_source: pd.DataFrame | None,
    player_ml: pd.DataFrame,
) -> pd.DataFrame | None:
    manual_xi = expand_manual_starting_xi(manual_xi_source)
    if manual_xi is None or manual_xi.empty:
        return None

    players = add_player_selection_score(player_ml)
    players["position_group"] = combined_position(players)
    player_cols = [
        "nation_code",
        "player_key",
        "position_group",
        "cur_squad",
        "prev_squad",
        "selection_score",
        "weighted_goal_form",
        "weighted_chance_form",
        "weighted_minutes",
        "cur_stat_source",
        "cur_save_pct",
        "cur_goals_per90",
        "cur_assists_per90",
        "cur_xg_per90",
        "cur_interceptions_per90",
        "cur_tackles_won_per90",
    ]
    player_cols = [col for col in player_cols if col in players.columns]
    stats = players[player_cols].drop_duplicates(["nation_code", "player_key"])
    out = manual_xi.merge(stats, on=["nation_code", "player_key"], how="left")
    out["selection_method"] = "manual_starting_xi"
    out["has_player_stats"] = out["weighted_minutes"].notna() if "weighted_minutes" in out.columns else False
    out = impute_manual_starter_stats(out, players)

    keep_cols = [
        "nation_code",
        "nation_name",
        "formation",
        "selection_method",
        "xi_order",
        "xi_role",
        "player_name",
        "player_key",
        "has_player_stats",
        "position_group",
        "cur_squad",
        "prev_squad",
        "selection_score",
        "weighted_goal_form",
        "weighted_chance_form",
        "weighted_minutes",
        "cur_stat_source",
        "cur_save_pct",
        "cur_goals_per90",
        "cur_assists_per90",
        "cur_xg_per90",
        "cur_interceptions_per90",
        "cur_tackles_won_per90",
    ]
    keep_cols = [col for col in keep_cols if col in out.columns]
    return out[keep_cols].sort_values(["nation_code", "xi_order"])


def manual_role_band(role: object) -> str:
    value = str(role).upper().strip()
    if value == "GK":
        return "GK"
    if value in {"RB", "CB", "LB", "RWB", "LWB", "RCB", "LCB"}:
        return "DF"
    if value in {"DM", "CM", "AM", "RM", "LM"}:
        return "MF"
    return "FW"


def player_position_band(position: object) -> str:
    value = str(position).upper()
    if "GK" in value:
        return "GK"
    if "DF" in value:
        return "DF"
    if "MF" in value:
        return "MF"
    if "FW" in value:
        return "FW"
    return "UNK"


def role_default_table(players: pd.DataFrame) -> pd.DataFrame:
    pool = players.copy()
    pool["role_band"] = pool["position_group"].map(player_position_band)
    pool = pool[pool["role_band"].isin(["GK", "DF", "MF", "FW"])].copy()

    defaults = (
        pool.groupby("role_band", as_index=True)
        .agg(
            selection_score=("selection_score", "median"),
            weighted_goal_form=("weighted_goal_form", "median"),
            weighted_chance_form=("weighted_chance_form", "median"),
            weighted_minutes=("weighted_minutes", "median"),
            cur_save_pct=("cur_save_pct", "median"),
            cur_goals_per90=("cur_goals_per90", "median"),
            cur_assists_per90=("cur_assists_per90", "median"),
            cur_xg_per90=("cur_xg_per90", "median"),
            cur_interceptions_per90=("cur_interceptions_per90", "median"),
            cur_tackles_won_per90=("cur_tackles_won_per90", "median"),
        )
    )

    if "GK" in defaults.index:
        gk_save = numeric(pool[pool["role_band"] == "GK"], "cur_save_pct")
        defaults.loc["GK", "cur_save_pct"] = float(gk_save[gk_save > 0].median()) if (gk_save > 0).any() else 68.0
    return defaults


def impute_manual_starter_stats(manual_xi: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    out = manual_xi.copy()
    defaults = role_default_table(players)
    out["manual_role_band"] = out["xi_role"].map(manual_role_band)

    fill_cols = [
        "selection_score",
        "weighted_goal_form",
        "weighted_chance_form",
        "weighted_minutes",
        "cur_save_pct",
        "cur_goals_per90",
        "cur_assists_per90",
        "cur_xg_per90",
        "cur_interceptions_per90",
        "cur_tackles_won_per90",
    ]
    for col in fill_cols:
        if col not in out.columns:
            out[col] = np.nan

    missing_stats = ~out["has_player_stats"].astype(bool)
    for idx, row in out[missing_stats].iterrows():
        band = row["manual_role_band"]
        if band not in defaults.index:
            continue
        for col in fill_cols:
            if pd.isna(out.at[idx, col]):
                out.at[idx, col] = defaults.at[band, col]
        if not str(out.at[idx, "cur_stat_source"]).strip() or pd.isna(out.at[idx, "cur_stat_source"]):
            out.at[idx, "cur_stat_source"] = "manual_xi_role_estimate"
        if not str(out.at[idx, "position_group"]).strip() or pd.isna(out.at[idx, "position_group"]):
            out.at[idx, "position_group"] = band

    for col in fill_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def aggregate_manual_xi_features(manual_starting_xi: pd.DataFrame | None) -> pd.DataFrame | None:
    if manual_starting_xi is None or manual_starting_xi.empty:
        return None

    rows = []
    for (nation_code, nation_name), xi in manual_starting_xi.groupby(["nation_code", "nation_name"], dropna=False):
        xi = xi.copy()
        roles = text(xi, "xi_role").str.upper()
        attackers = xi[roles.isin(["RW", "LW", "ST", "CF", "SS"])]
        creators = xi[roles.isin(["AM", "RW", "LW", "CM", "DM"])]
        defenders = xi[roles.isin(["RB", "LB", "CB", "RWB", "LWB"])]
        keepers = xi[roles.eq("GK")]
        xi_count = len(xi)
        stats_count = int(xi["has_player_stats"].sum()) if "has_player_stats" in xi.columns else 0

        rows.append(
            {
                "nation_code": nation_code,
                "nation_name": nation_name,
                "manual_xi_players": xi_count,
                "manual_xi_players_with_stats": stats_count,
                "manual_xi_coverage_pct": stats_count / xi_count if xi_count else 0.0,
                "manual_xi_goal_form_sum": top_n_sum(attackers, "weighted_goal_form", 4),
                "manual_xi_goal_form_mean": float(numeric(attackers, "weighted_goal_form").mean()) if len(attackers) else 0.0,
                "manual_xi_chance_form_sum": top_n_sum(creators, "weighted_chance_form", 6),
                "manual_xi_chance_form_mean": float(numeric(creators, "weighted_chance_form").mean()) if len(creators) else 0.0,
                "manual_xi_defender_actions_mean": float(
                    (
                        numeric(defenders, "cur_interceptions_per90")
                        + numeric(defenders, "cur_tackles_won_per90")
                    ).mean()
                )
                if len(defenders)
                else 0.0,
                "manual_xi_keeper_save_pct_best": float(numeric(keepers, "cur_save_pct").max()) if len(keepers) else 0.0,
                "manual_xi_weighted_minutes_mean": float(numeric(xi, "weighted_minutes").mean()) if len(xi) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def add_manual_team_placeholders(
    probable_squads: pd.DataFrame,
    manual_xi_features: pd.DataFrame | None,
) -> pd.DataFrame:
    if manual_xi_features is None or manual_xi_features.empty:
        return probable_squads

    existing = set(text(probable_squads, "nation_code"))
    needed = manual_xi_features[~manual_xi_features["nation_code"].isin(existing)]
    if needed.empty:
        return probable_squads

    numeric_defaults = {
        col: float(pd.to_numeric(probable_squads[col], errors="coerce").quantile(0.20))
        for col in [
            "avg_selection_score",
            "top6_goal_form_sum",
            "top6_goal_form_mean",
            "top10_chance_form_sum",
            "top10_chance_form_mean",
            "attacker_goal_form_mean",
            "midfielder_chance_form_mean",
            "defender_actions_per90_mean",
            "keeper_save_pct_best",
            "squad_weighted_minutes_mean",
        ]
        if col in probable_squads.columns
    }

    rows = []
    for _, team in needed.iterrows():
        row = {
            "nation_code": team["nation_code"],
            "nation_name": team["nation_name"],
            "selection_method": "manual_xi_missing_player_pool",
            "squad_players_used": 0,
            "estimated_players_used": 0,
            "verified_supplemental_used": 0,
        }
        row.update(numeric_defaults)
        rows.append(row)

    print(f"Added manual-XI placeholder teams: {', '.join(needed['nation_code'].astype(str))}")
    return pd.concat([probable_squads, pd.DataFrame(rows)], ignore_index=True, sort=False)


def build_power_scores(
    probable_squads: pd.DataFrame,
    squad_coverage: pd.DataFrame | None = None,
    manual_xi_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = add_manual_team_placeholders(probable_squads, manual_xi_features).copy()
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

    if manual_xi_features is not None and not manual_xi_features.empty:
        df = df.merge(
            manual_xi_features.drop(columns=["nation_name"], errors="ignore"),
            on="nation_code",
            how="left",
        )
        xi_coverage = numeric(df, "manual_xi_coverage_pct")
        has_manual_xi = numeric(df, "manual_xi_players") >= 11

        manual_attack_score = (
            0.65 * percentile_score(numeric(df, "manual_xi_goal_form_sum"))
            + 0.35 * percentile_score(numeric(df, "manual_xi_goal_form_mean"))
        )
        manual_creation_score = (
            0.65 * percentile_score(numeric(df, "manual_xi_chance_form_sum"))
            + 0.35 * percentile_score(numeric(df, "manual_xi_chance_form_mean"))
        )
        manual_defense_score = (
            0.60 * percentile_score(numeric(df, "manual_xi_defender_actions_mean"))
            + 0.40 * percentile_score(numeric(df, "manual_xi_weighted_minutes_mean"))
        )
        manual_keeper_score = percentile_score(numeric(df, "manual_xi_keeper_save_pct_best"))

        df["attack_score"] = np.where(
            has_manual_xi,
            0.70 * df["attack_score"] + 0.30 * manual_attack_score,
            df["attack_score"],
        )
        df["creation_score"] = np.where(
            has_manual_xi,
            0.75 * df["creation_score"] + 0.25 * manual_creation_score,
            df["creation_score"],
        )
        df["defense_score"] = np.where(
            has_manual_xi,
            0.85 * df["defense_score"] + 0.15 * manual_defense_score,
            df["defense_score"],
        )
        df["keeper_score"] = np.where(
            has_manual_xi,
            0.20 * df["keeper_score"] + 0.80 * manual_keeper_score,
            df["keeper_score"],
        )
    else:
        xi_coverage = pd.Series(1.0, index=df.index)

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
    df["data_confidence_score"] = df["data_confidence_score"] * (0.90 + 0.10 * xi_coverage.clip(0, 1))
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
    manual_xi_source = read_manual_starting_xi_source()
    squad_coverage_path = PROCESSED_DIR / "squad_nation_coverage.csv"
    squad_coverage = pd.read_csv(squad_coverage_path) if squad_coverage_path.exists() else None
    squad_player_path = PROCESSED_DIR / "squad_player_coverage.csv"
    squad_player_coverage = pd.read_csv(squad_player_path) if squad_player_path.exists() else None

    team_squads = aggregate_team_squads(player_ml, squad_player_coverage)
    selected_squad_players = select_team_squad_players(player_ml, squad_player_coverage)
    generated_starting_xi = build_probable_starting_xi(selected_squad_players)
    manual_starting_xi = build_manual_starting_xi(manual_xi_source, player_ml)
    manual_xi_features = aggregate_manual_xi_features(manual_starting_xi)
    probable_starting_xi = manual_starting_xi if manual_starting_xi is not None else generated_starting_xi
    final_features = build_power_scores(team_squads, squad_coverage, manual_xi_features)

    probable_path = PROCESSED_DIR / "team_squad_features.csv"
    selected_path = PROCESSED_DIR / "selected_squad_players.csv"
    xi_path = PROCESSED_DIR / "probable_starting_xi.csv"
    generated_xi_path = PROCESSED_DIR / "generated_starting_xi.csv"
    manual_xi_path = PROCESSED_DIR / "manual_starting_xi_expanded.csv"
    manual_xi_features_path = PROCESSED_DIR / "manual_xi_features.csv"
    final_path = PROCESSED_DIR / "final_team_features.csv"
    team_squads.to_csv(probable_path, index=False)
    selected_squad_players.to_csv(selected_path, index=False)
    generated_starting_xi.to_csv(generated_xi_path, index=False)
    if manual_starting_xi is not None:
        manual_starting_xi.to_csv(manual_xi_path, index=False)
    if manual_xi_features is not None:
        manual_xi_features.to_csv(manual_xi_features_path, index=False)
    probable_starting_xi.to_csv(xi_path, index=False)
    final_features.to_csv(final_path, index=False)

    print(f"team_squad_features: {team_squads.shape} -> {probable_path}")
    print(f"selected_squad_players: {selected_squad_players.shape} -> {selected_path}")
    print(f"generated_starting_xi: {generated_starting_xi.shape} -> {generated_xi_path}")
    if manual_starting_xi is not None:
        print(f"manual_starting_xi_expanded: {manual_starting_xi.shape} -> {manual_xi_path}")
    if manual_xi_features is not None:
        print(f"manual_xi_features: {manual_xi_features.shape} -> {manual_xi_features_path}")
    print(f"probable_starting_xi: {probable_starting_xi.shape} -> {xi_path}")
    print(f"final_team_features: {final_features.shape} -> {final_path}")
    print(final_features.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
