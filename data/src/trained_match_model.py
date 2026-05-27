from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


NAME_TO_CODE = {
    "Algeria": "DZA",
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA",
    "Cabo Verde": "CPV",
    "Cape Verde": "CPV",
    "Canada": "CAN",
    "Colombia": "COL",
    "Costa Rica": "CRC",
    "Croatia": "CRO",
    "Curacao": "CUW",
    "Curaçao": "CUW",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "DR Congo": "COD",
    "Democratic Republic of Congo": "COD",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Haiti": "HAI",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ivory Coast": "CIV",
    "Cote d'Ivoire": "CIV",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NED",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Panama": "PAN",
    "Paraguay": "PAR",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Saudi Arabia": "SAU",
    "Scotland": "SCO",
    "Senegal": "SEN",
    "South Africa": "RSA",
    "South Korea": "KOR",
    "Korea Republic": "KOR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Türkiye": "TUR",
    "United States": "USA",
    "USA": "USA",
    "Uruguay": "URU",
    "Uzbekistan": "UZB",
}


MODEL_SCORE_COLUMNS = [
    "power_score",
    "attack_score",
    "creation_score",
    "defense_score",
    "keeper_score",
    "depth_score",
    "data_confidence_score",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams = pd.read_csv(PROCESSED_DIR / "final_team_features.csv", encoding="latin-1")
    results = pd.read_csv(RAW_DIR / "results.csv", encoding="latin-1")
    fixtures = pd.read_csv(RAW_DIR / "wc_2026_group_fixtures.csv", encoding="latin-1")

    print(f"teams: {teams.shape}")
    print(list(teams.columns))
    print(f"results: {results.shape}")
    print(list(results.columns))
    print(f"fixtures: {fixtures.shape}")
    print(list(fixtures.columns))
    return teams, results, fixtures


def add_missing_team_placeholders(teams: pd.DataFrame, needed_codes: set[str]) -> pd.DataFrame:
    existing = set(teams["nation_code"])
    missing = sorted(needed_codes - existing)
    if not missing:
        return teams

    low_values = {
        col: float(pd.to_numeric(teams[col], errors="coerce").quantile(0.20))
        for col in MODEL_SCORE_COLUMNS
        if col in teams.columns
    }
    rows = []
    reverse_names = {code: name for name, code in NAME_TO_CODE.items()}
    for code in missing:
        row = {
            "nation_code": code,
            "nation_name": reverse_names.get(code, code),
            "selection_method": "missing_data_placeholder",
            "squad_players_used": 0,
            "estimated_players_used": 0,
            "verified_supplemental_used": 0,
            "coverage_pct": 0,
            "has_announced_squad_file": False,
        }
        row.update(low_values)
        row["data_confidence_score"] = min(row.get("data_confidence_score", 20), 20)
        rows.append(row)

    print(f"Added missing-data placeholders for: {missing}")
    return pd.concat([teams, pd.DataFrame(rows)], ignore_index=True, sort=False)


def prepare_results(results: pd.DataFrame, min_year: int = 2018) -> pd.DataFrame:
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed", dayfirst=True)
    df = df[df["date"].dt.year >= min_year].copy()
    df = df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])
    df["home_code"] = df["home_team"].map(NAME_TO_CODE)
    df["away_code"] = df["away_team"].map(NAME_TO_CODE)
    df = df.dropna(subset=["home_code", "away_code"])

    df["result"] = np.select(
        [
            df["home_score"] > df["away_score"],
            df["home_score"] == df["away_score"],
            df["home_score"] < df["away_score"],
        ],
        ["home_win", "draw", "away_win"],
        default="unknown",
    )
    df = df[df["result"] != "unknown"].copy()
    return df


def team_lookup(teams: pd.DataFrame) -> dict[str, pd.Series]:
    return {row["nation_code"]: row for _, row in teams.iterrows()}


def matchup_features(team_a: pd.Series, team_b: pd.Series) -> dict[str, float]:
    features = {}
    for col in MODEL_SCORE_COLUMNS:
        a = float(team_a.get(col, 0) or 0)
        b = float(team_b.get(col, 0) or 0)
        features[f"{col}_diff"] = a - b
        features[f"{col}_absdiff"] = abs(a - b)
    features["attack_vs_defense_diff"] = float(team_a.get("attack_score", 0)) - float(team_b.get("defense_score", 0))
    features["defense_vs_attack_diff"] = float(team_a.get("defense_score", 0)) - float(team_b.get("attack_score", 0))
    return features


def build_training_data(results: pd.DataFrame, teams: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    lookup = team_lookup(teams)
    rows = []
    labels = []

    for _, match in results.iterrows():
        home = lookup.get(match["home_code"])
        away = lookup.get(match["away_code"])
        if home is None or away is None:
            continue
        rows.append(matchup_features(home, away))
        labels.append(match["result"])

    X = pd.DataFrame(rows).fillna(0)
    y = pd.Series(labels, name="result")
    return X, y


def train_models(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, RandomForestClassifier, pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced_subsample",
    )

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    rows = []
    for name, model in [("logistic_regression", logistic), ("random_forest", forest)]:
        pred = model.predict(X_test)
        rows.append({"model": name, "accuracy": accuracy_score(y_test, pred)})
        print(f"\n{name}")
        print(classification_report(y_test, pred))

    metrics = pd.DataFrame(rows)
    return logistic, forest, metrics


def predict_fixtures(fixtures: pd.DataFrame, teams: pd.DataFrame, model) -> pd.DataFrame:
    lookup = team_lookup(teams)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["model"].classes_)

    rows = []
    for _, fixture in fixtures.iterrows():
        team_a_code = NAME_TO_CODE.get(fixture["team_a"])
        team_b_code = NAME_TO_CODE.get(fixture["team_b"])
        team_a = lookup.get(team_a_code)
        team_b = lookup.get(team_b_code)
        if team_a is None or team_b is None:
            rows.append({**fixture.to_dict(), "prediction_status": "missing_team_features"})
            continue

        X = pd.DataFrame([matchup_features(team_a, team_b)]).fillna(0)
        probs = dict(zip(classes, model.predict_proba(X)[0]))
        predicted = max(probs, key=probs.get)
        rows.append(
            {
                **fixture.to_dict(),
                "team_a_code": team_a_code,
                "team_b_code": team_b_code,
                "prediction_status": "ok",
                "team_a_win_prob": probs.get("home_win", 0),
                "draw_prob": probs.get("draw", 0),
                "team_b_win_prob": probs.get("away_win", 0),
                "predicted_result": predicted,
            }
        )

    return pd.DataFrame(rows)


def build_group_tables(predictions: pd.DataFrame) -> pd.DataFrame:
    table = {}
    for _, match in predictions[predictions["prediction_status"] == "ok"].iterrows():
        group = match["group"]
        a = match["team_a"]
        b = match["team_b"]
        for team in [a, b]:
            table.setdefault(
                (group, team),
                {"group": group, "team": team, "points": 0.0, "gf_x": 0.0, "ga_x": 0.0},
            )

        a_pts = 3 * match["team_a_win_prob"] + match["draw_prob"]
        b_pts = 3 * match["team_b_win_prob"] + match["draw_prob"]
        table[(group, a)]["points"] += a_pts
        table[(group, b)]["points"] += b_pts
        table[(group, a)]["gf_x"] += match["team_a_win_prob"] * 2 + match["draw_prob"]
        table[(group, b)]["gf_x"] += match["team_b_win_prob"] * 2 + match["draw_prob"]
        table[(group, a)]["ga_x"] += match["team_b_win_prob"] * 2 + match["draw_prob"]
        table[(group, b)]["ga_x"] += match["team_a_win_prob"] * 2 + match["draw_prob"]

    df = pd.DataFrame(table.values())
    df["gd_x"] = df["gf_x"] - df["ga_x"]
    return df.sort_values(["group", "points", "gd_x", "gf_x"], ascending=[True, False, False, False])


def main() -> None:
    teams, results, fixtures = load_inputs()
    fixture_codes = set(fixtures["team_a"].map(NAME_TO_CODE).dropna()) | set(fixtures["team_b"].map(NAME_TO_CODE).dropna())
    teams = add_missing_team_placeholders(teams, fixture_codes)
    recent_results = prepare_results(results, min_year=2018)
    X, y = build_training_data(recent_results, teams)

    print(f"Training matches after mapping: {len(X)}")
    print(f"Target distribution:\n{y.value_counts()}")

    logistic, forest, metrics = train_models(X, y)
    best_model_name = metrics.sort_values("accuracy", ascending=False).iloc[0]["model"]
    best_model = logistic if best_model_name == "logistic_regression" else forest
    print(f"\nUsing best validation model for fixtures: {best_model_name}")

    predictions = predict_fixtures(fixtures, teams, best_model)
    group_tables = build_group_tables(predictions)

    metrics.to_csv(PROCESSED_DIR / "trained_model_metrics.csv", index=False)
    predictions.to_csv(PROCESSED_DIR / "wc_2026_group_predictions.csv", index=False)
    group_tables.to_csv(PROCESSED_DIR / "wc_2026_predicted_group_tables.csv", index=False)

    print("\nModel metrics:")
    print(metrics.to_string(index=False))
    print("\nGroup prediction sample:")
    print(predictions.head(12).to_string(index=False))
    print("\nPredicted group tables:")
    print(group_tables.to_string(index=False))


if __name__ == "__main__":
    main()
