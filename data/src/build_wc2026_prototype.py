from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANUAL_DIR = PROJECT_ROOT / "data" / "manual"
ANNEX_C_PATH = MANUAL_DIR / "fifa_annex_c_third_place_allocations.csv"
OUT_PATH = PROJECT_ROOT / "wc2026_prediction_experience.html"
FRESH_OUT_PATH = PROJECT_ROOT / "wc2026_prototype_fresh.html"


FLAGS = {
    "ARG": "ar", "FRA": "fr", "ENG": "gb-eng", "BEL": "be", "GER": "de", "POR": "pt",
    "NED": "nl", "ESP": "es", "SEN": "sn", "BRA": "br", "SUI": "ch", "AUT": "at",
    "NOR": "no", "COL": "co", "CRO": "hr", "SWE": "se", "MAR": "ma", "USA": "us",
    "TUR": "tr", "BIH": "ba", "CZE": "cz", "PAR": "py", "EGY": "eg", "IRN": "ir",
    "CIV": "ci", "NZL": "nz", "MEX": "mx", "URU": "uy", "KOR": "kr", "COD": "cd",
    "JPN": "jp", "GHA": "gh", "HAI": "ht", "QAT": "qa", "UZB": "uz", "CUW": "cw",
    "CPV": "cv", "PAN": "pa", "SCO": "gb-sct", "JOR": "jo", "RSA": "za", "TUN": "tn",
    "CAN": "ca", "ECU": "ec", "IRQ": "iq", "DZA": "dz", "SAU": "sa", "AUS": "au",
}


def clean_number(value: object, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def load_data() -> tuple[list[dict], list[dict], list[dict], dict, dict, dict]:
    teams = pd.read_csv(PROCESSED_DIR / "final_team_features.csv")
    title = pd.read_csv(PROCESSED_DIR / "wc_2026_monte_carlo_title_probabilities.csv")
    group_predictions = pd.read_csv(PROCESSED_DIR / "wc_2026_group_predictions.csv")
    group_tables = pd.read_csv(PROCESSED_DIR / "wc_2026_predicted_group_tables.csv")
    manual_xi = pd.read_csv(MANUAL_DIR / "manual_starting_xi.csv")
    annex_c = pd.read_csv(ANNEX_C_PATH, dtype=str)

    title_map = title.set_index("nation_code").to_dict("index")
    team_rows = []
    for _, row in teams.iterrows():
        code = row["nation_code"]
        title_row = title_map.get(code, {})
        team_rows.append(
            {
                "code": code,
                "name": row["nation_name"],
                "flag": FLAGS.get(code, ""),
                "power": clean_number(row.get("power_score"), 2),
                "attack": clean_number(row.get("attack_score"), 2),
                "creation": clean_number(row.get("creation_score"), 2),
                "defense": clean_number(row.get("defense_score"), 2),
                "keeper": clean_number(row.get("keeper_score"), 2),
                "depth": clean_number(row.get("depth_score"), 2),
                "confidence": clean_number(row.get("data_confidence_score"), 2),
                "titleProb": clean_number(title_row.get("title_prob", 0), 4),
                "finalProb": clean_number(title_row.get("final_prob", 0), 4),
                "semiProb": clean_number(title_row.get("semifinal_prob", 0), 4),
                "qfProb": clean_number(title_row.get("quarterfinal_prob", 0), 4),
                "r16Prob": clean_number(title_row.get("round_of_16_prob", 0), 4),
            }
        )

    title_rows = [
        {
            "code": row["nation_code"],
            "name": row["nation_name"],
            "titleProb": clean_number(row["title_prob"], 4),
            "finalProb": clean_number(row["final_prob"], 4),
            "semiProb": clean_number(row["semifinal_prob"], 4),
            "qfProb": clean_number(row["quarterfinal_prob"], 4),
            "r16Prob": clean_number(row["round_of_16_prob"], 4),
        }
        for _, row in title.iterrows()
    ]

    fixture_name_to_code = {}
    match_rows = []
    for _, row in group_predictions.iterrows():
        fixture_name_to_code[row["team_a"]] = row["team_a_code"]
        fixture_name_to_code[row["team_b"]] = row["team_b_code"]
        match_rows.append(
            {
                "stage": "Group",
                "group": row["group"],
                "match": f"M{int(row['match_number'])}",
                "date": row["date"],
                "venue": row["venue"],
                "a": row["team_a"],
                "aCode": row["team_a_code"],
                "b": row["team_b"],
                "bCode": row["team_b_code"],
                "aWin": clean_number(row["team_a_win_prob"], 4),
                "draw": clean_number(row["draw_prob"], 4),
                "bWin": clean_number(row["team_b_win_prob"], 4),
                "pick": row["predicted_result"],
            }
        )

    aliases = {
        "Cape Verde": "CPV",
        "Czech Republic": "CZE",
        "Czechia": "CZE",
        "Ivory Coast": "CIV",
        "Cote d'Ivoire": "CIV",
        "Curacao": "CUW",
        "Curaçao": "CUW",
        "Türkiye": "TUR",
        "Turkey": "TUR",
        "United States": "USA",
        "South Korea": "KOR",
        "DR Congo": "COD",
        "Bosnia and Herzegovina": "BIH",
    }
    feature_name_to_code = {team["name"]: team["code"] for team in team_rows}
    tables: dict[str, list[dict]] = {}
    for group, group_df in group_tables.groupby("group"):
        rows = []
        for _, row in group_df.sort_values(["points", "gd_x", "gf_x"], ascending=False).iterrows():
            code = fixture_name_to_code.get(row["team"]) or feature_name_to_code.get(row["team"]) or aliases.get(row["team"])
            if not code:
                raise ValueError(f"Could not map group-table team to nation code: {row['team']}")
            rows.append(
                {
                    "team": row["team"],
                    "code": code,
                    "points": clean_number(row["points"], 2),
                    "gd": clean_number(row["gd_x"], 2),
                    "gf": clean_number(row["gf_x"], 2),
                }
            )
        tables[str(group)] = rows

    lineups = {}
    for _, row in manual_xi.iterrows():
        players = []
        for item in str(row["xi"]).split(";"):
            if ":" not in item:
                continue
            role, name = item.split(":", 1)
            players.append({"role": role.strip(), "name": name.strip()})
        lineups[row["nation_code"]] = {"formation": row["formation"], "players": players}

    slot_columns = {
        "M79": "M79_1A",
        "M85": "M85_1B",
        "M81": "M81_1D",
        "M74": "M74_1E",
        "M82": "M82_1G",
        "M77": "M77_1I",
        "M87": "M87_1K",
        "M80": "M80_1L",
    }
    annex_lookup = {
        row["qualified_groups"]: {match_id: row[column] for match_id, column in slot_columns.items()}
        for _, row in annex_c.iterrows()
    }

    return team_rows, title_rows, match_rows, tables, lineups, annex_lookup


def build_html() -> str:
    teams, title_rows, matches, tables, lineups, annex_lookup = load_data()
    data_script = (
        f"const TEAMS={json.dumps(teams, ensure_ascii=False)};\n"
        f"const TITLE={json.dumps(title_rows, ensure_ascii=False)};\n"
        f"const GROUP_MATCHES={json.dumps(matches, ensure_ascii=False)};\n"
        f"const GROUP_TABLES={json.dumps(tables, ensure_ascii=False)};\n"
        f"const LINEUPS={json.dumps(lineups, ensure_ascii=False)};\n"
        f"const ANNEX_C={json.dumps(annex_lookup, ensure_ascii=False)};\n"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WC2026 Predictor Prototype</title>
<style>
:root{{--bg:#06100e;--panel:#0d1714;--panel2:#13231e;--line:#27483f;--line2:#3f6659;--text:#f2fbf7;--muted:#9fb8af;--gold:#f7c948;--green:#37d47f;--blue:#58a6ff;--red:#ff6b6b;--orange:#ff9f43;--ink:#07100d}}html{{min-height:100%;background:#040707}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 18% 0,#16392e 0,#07110f 38%,#040707 100%);font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--text)}}button,input{{font:inherit}}button{{cursor:pointer;color:inherit}}.app{{min-height:100vh}}.top{{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:14px;padding:13px 20px;background:rgba(5,10,9,.88);backdrop-filter:blur(16px);border-bottom:1px solid rgba(255,255,255,.08)}}.brand{{display:flex;align-items:center;gap:10px;font-weight:950;letter-spacing:.08em;white-space:nowrap}}.brand-mark{{width:34px;height:34px;border-radius:12px;background:var(--gold);color:#161109;display:grid;place-items:center}}.nav{{display:flex;gap:8px;flex:1;overflow:auto;scrollbar-width:none}}.nav::-webkit-scrollbar{{display:none}}.nav button,.sound,.ghost,.chip{{border:1px solid var(--line);background:#0b1512;border-radius:999px;padding:9px 13px;font-weight:850}}.nav button.active,.chip.active{{background:var(--gold);border-color:var(--gold);color:#151109}}.sound{{min-width:88px}}.wrap{{max-width:1420px;margin:0 auto;padding:22px}}.page{{display:none}}.page.active{{display:block}}.hero{{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;align-items:stretch}}.hero-main,.panel,.team-card,.match-card{{background:linear-gradient(180deg,rgba(19,35,30,.96),rgba(9,18,15,.96));border:1px solid rgba(255,255,255,.08);border-radius:18px;box-shadow:0 18px 46px rgba(0,0,0,.25)}}.hero-main{{padding:30px;position:relative;overflow:hidden;min-height:430px;display:grid;align-content:center}}.hero-main:after{{content:"";position:absolute;right:-110px;bottom:-120px;width:380px;height:380px;border-radius:50%;background:radial-gradient(circle,rgba(247,201,72,.28),transparent 65%)}}.hero-main>*{{position:relative;z-index:1}}.kicker{{color:var(--gold);font-size:12px;font-weight:950;letter-spacing:.16em;text-transform:uppercase}}h1{{font-size:clamp(44px,7vw,92px);line-height:.92;letter-spacing:0;margin:12px 0 14px}}h2{{margin:0;font-size:24px}}h3{{margin:0;font-size:17px}}p{{margin:0}}.sub{{color:var(--muted);line-height:1.55;max-width:780px}}.actions{{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}}.primary{{border:0;background:linear-gradient(135deg,var(--gold),#fff1a8);color:#171209;border-radius:14px;padding:15px 18px;font-weight:950;box-shadow:0 12px 26px rgba(247,201,72,.22)}}.ghost{{border-radius:14px;padding:13px 16px}}.panel{{padding:18px}}.section-head{{display:flex;align-items:end;justify-content:space-between;gap:12px;margin-bottom:14px}}.small{{color:var(--muted);font-size:12px;line-height:1.45}}.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px;margin-top:18px}}.span4{{grid-column:span 4}}.span5{{grid-column:span 5}}.span7{{grid-column:span 7}}.span8{{grid-column:span 8}}.span12{{grid-column:span 12}}.podium{{display:grid;gap:12px;height:100%}}.winner-card{{display:grid;grid-template-columns:auto auto 1fr auto;gap:12px;align-items:center;padding:14px;border-radius:16px;background:#0a1411;border:1px solid rgba(255,255,255,.08)}}.winner-card:first-child{{border-color:rgba(247,201,72,.65);background:linear-gradient(90deg,#1e2113,#0a1411)}}.rank{{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;background:var(--gold);color:#151109;font-weight:950}}.flag{{width:38px;height:26px;object-fit:cover;border-radius:5px;box-shadow:0 0 0 1px rgba(255,255,255,.2)}}.flagbox{{width:38px;height:26px;border-radius:5px;background:#243a33;display:grid;place-items:center;font-size:10px;font-weight:950}}.teamline{{display:flex;align-items:center;gap:10px;min-width:0}}.teamline b{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.probbar{{height:9px;background:#23352f;border-radius:999px;overflow:hidden}}.probbar i{{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--green),var(--gold))}}.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}}.match-list{{display:grid;grid-template-columns:repeat(auto-fill,minmax(305px,1fr));gap:12px}}.match-card{{padding:13px;text-align:left;display:grid;gap:10px}}.match-top{{display:flex;justify-content:space-between;gap:10px;color:var(--muted);font-size:12px}}.versus{{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);gap:10px;align-items:center}}.versus .teamline:last-child{{justify-content:end;text-align:right}}.score{{color:var(--gold);font-size:20px;font-weight:950;min-width:58px;text-align:center}}.mini-probs{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px}}.mini-probs span{{height:5px;background:#263a33;border-radius:99px;overflow:hidden}}.mini-probs i{{display:block;height:100%;background:var(--green)}}.team-tools{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}}.team-tools input{{min-width:250px;flex:1;border:1px solid var(--line);background:#07100d;color:var(--text);border-radius:14px;padding:12px 14px;outline:none}}.team-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}}.team-card{{padding:15px;text-align:left;transition:.18s}}.team-card:hover{{transform:translateY(-2px);border-color:rgba(247,201,72,.55)}}.rings{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}}.ring-box{{text-align:center}}.ring{{--p:50;--c:var(--green);width:66px;height:66px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--c) calc(var(--p)*1%),#253730 0);position:relative;margin:auto}}.ring:after{{content:"";position:absolute;inset:7px;border-radius:50%;background:#0a1411}}.ring b{{position:relative;z-index:1;font-size:13px}}.ring-label{{font-size:10px;color:var(--muted);margin-top:4px}}.detail-layout{{display:grid;grid-template-columns:390px minmax(0,1fr);gap:18px}}.detail-card{{position:sticky;top:86px;align-self:start}}.detail-title{{display:flex;align-items:center;gap:12px;margin-bottom:14px}}.detail-title .flag,.detail-title .flagbox{{width:60px;height:40px}}.detail-title h2{{font-size:34px}}.radar{{width:100%;max-width:330px;margin:12px auto;display:block}}.stat-lines{{display:grid;gap:9px;margin-top:12px}}.stat-line{{display:grid;grid-template-columns:92px 1fr 44px;gap:8px;align-items:center;font-size:12px}}.bar{{height:8px;border-radius:99px;background:#23352f;overflow:hidden}}.bar i{{display:block;height:100%;background:linear-gradient(90deg,var(--green),var(--gold))}}.lineup-shell{{display:grid;grid-template-columns:280px minmax(0,1fr);gap:16px}}.lineup-list{{display:grid;gap:8px;max-height:650px;overflow:auto;padding-right:4px}}.lineup-list button{{border:1px solid transparent;background:#0a1411;border-radius:13px;padding:10px;text-align:left}}.lineup-list button.active,.lineup-list button:hover{{border-color:var(--gold);background:#101b17}}.field{{height:650px;border-radius:20px;background:linear-gradient(90deg,#145233,#1d7449);border:2px solid rgba(255,255,255,.45);position:relative;overflow:hidden}}.field:before{{content:"";position:absolute;inset:20px;border:2px solid rgba(255,255,255,.34);border-radius:14px}}.field:after{{content:"";position:absolute;left:50%;top:50%;width:140px;height:140px;border:2px solid rgba(255,255,255,.32);border-radius:50%;transform:translate(-50%,-50%)}}.half{{position:absolute;left:20px;right:20px;top:50%;border-top:2px solid rgba(255,255,255,.3)}}.box{{position:absolute;left:28%;right:28%;height:86px;border:2px solid rgba(255,255,255,.26)}}.box.top{{top:20px;border-top:0}}.box.bottom{{bottom:20px;border-bottom:0}}.player{{position:absolute;transform:translate(-50%,-50%);text-align:center;z-index:2;width:116px}}.disc{{width:44px;height:44px;border-radius:50%;display:grid;place-items:center;margin:auto;background:var(--gold);color:#151109;font-weight:950;box-shadow:0 10px 20px rgba(0,0,0,.32)}}.player span{{display:block;margin-top:5px;font-size:11px;line-height:1.08;font-weight:850;text-shadow:0 2px 8px #000}}.player small{{display:block;font-size:9px;color:#e4f3ed}}.toast{{position:fixed;left:50%;bottom:18px;transform:translateX(-50%) translateY(20px);opacity:0;background:#0a1411;border:1px solid rgba(247,201,72,.55);border-radius:16px;padding:13px 16px;transition:.2s;box-shadow:0 14px 40px rgba(0,0,0,.4);z-index:60}}.toast.show{{opacity:1;transform:translateX(-50%) translateY(0)}}
.player{{width:102px}}.disc{{width:40px;height:40px}}.player span{{margin-top:4px;font-size:10px;line-height:1.05;overflow-wrap:anywhere}}
@media(max-width:980px){{.hero,.detail-layout,.lineup-shell{{grid-template-columns:1fr}}.grid{{grid-template-columns:1fr}}.span4,.span5,.span7,.span8,.span12{{grid-column:span 1}}.detail-card{{position:static}}.field{{height:520px}}.wrap{{padding:14px}}}}
@media(max-width:620px){{.top{{flex-wrap:wrap}}.podium .winner-card,.winner-card{{grid-template-columns:auto 1fr auto}}.winner-card .flag,.winner-card .flagbox{{display:none}}.rings{{grid-template-columns:repeat(2,1fr)}}.match-list{{grid-template-columns:1fr}}h1{{font-size:46px}}}}
</style>
</head>
<body>
<audio id="bgMusic" src="assets/audio/dai-shakira-burna-boy.m4a" loop preload="auto"></audio>
<div class="app">
<header class="top">
  <div class="brand"><div class="brand-mark">26</div><span>WC PREDICTOR</span></div>
  <nav class="nav">
    <button class="active" data-page="home">Home</button>
    <button data-page="matches">Matches</button>
    <button data-page="teams">Teams</button>
    <button data-page="lineups">Lineups</button>
  </nav>
  <button class="sound" id="soundBtn">Play</button>
</header>
<main class="wrap">
  <section id="home" class="page active">
    <div class="hero">
      <div class="hero-main">
        <div class="kicker">10,000 simulations · strict FIFA bracket</div>
        <h1>Predict the World Cup winner.</h1>
        <p class="sub">Player-form features, manual probable XIs, team strengths, group-stage forecasts, and official FIFA Annex C third-place allocation in a prototype made for exploration.</p>
        <div class="actions"><button class="primary" id="predictBtn">Predict Winner</button><button class="ghost" onclick="showPage('matches')">View Matches</button><button class="ghost" onclick="showPage('teams')">Explore Teams</button></div>
      </div>
      <div class="panel"><div class="section-head"><h2>Projected Podium</h2><span class="small">Monte Carlo title path</span></div><div class="podium" id="podium"></div></div>
    </div>
    <div class="grid">
      <div class="panel span7"><div class="section-head"><h2>Title Probability</h2><span class="small">Top 12</span></div><div id="titleBars"></div></div>
      <div class="panel span5"><div class="section-head"><h2>Model Notes</h2><span class="small">Current version</span></div><div id="notes"></div></div>
    </div>
  </section>
  <section id="matches" class="page">
    <div class="section-head"><h2>Match Center</h2><span class="small">Group stage first, then R32, R16, QF, SF, Final</span></div>
    <div class="tabs" id="stageTabs"></div>
    <div id="matchOutput" class="match-list"></div>
  </section>
  <section id="teams" class="page">
    <div id="teamBrowse">
      <div class="section-head"><h2>All Nations</h2><span class="small">Click a team for full stats and XI</span></div>
      <div class="team-tools"><input id="teamSearch" placeholder="Search nation"><button class="ghost" onclick="sortTeams('title')">Sort Title %</button><button class="ghost" onclick="sortTeams('attack')">Sort Attack</button><button class="ghost" onclick="sortTeams('defense')">Sort Defense</button></div>
      <div id="teamGrid" class="team-grid"></div>
    </div>
    <div id="teamDetail" style="display:none"></div>
  </section>
  <section id="lineups" class="page">
    <div class="section-head"><h2>Predicted Starting XIs</h2><span class="small">Role-correct pitch view</span></div>
    <div class="lineup-shell"><div class="lineup-list" id="lineupList"></div><div id="lineupPitch"></div></div>
  </section>
</main>
</div>
<div class="toast" id="toast"></div>
<script>
{data_script}
const R32=[['M73','2A','2B'],['M76','1C','2F'],['M74','1E','3ABCDF'],['M75','1F','2C'],['M78','2E','2I'],['M77','1I','3CDFGH'],['M79','1A','3CEFHI'],['M80','1L','3EHIJK'],['M82','1G','3AEHIJ'],['M81','1D','3BEFIJ'],['M84','1H','2J'],['M83','2K','2L'],['M85','1B','3EFGIJ'],['M88','2D','2G'],['M86','1J','2H'],['M87','1K','3DEIJL']];
const R16=[['M90','M73','M75'],['M89','M74','M77'],['M91','M76','M78'],['M92','M79','M80'],['M93','M83','M84'],['M94','M81','M82'],['M95','M86','M88'],['M96','M85','M87']];
const QF=[['M97','M89','M90'],['M98','M93','M94'],['M99','M91','M92'],['M100','M95','M96']];
const SF=[['M101','M97','M98'],['M102','M99','M100']];
const FINAL=['M104','M101','M102'];
const THIRD_SLOT_ORDER=['M79','M85','M81','M74','M82','M77','M87','M80'];
const STAGES=['Group','Round of 32','Round of 16','Quarterfinal','Semifinal','Final'];
const byCode=Object.fromEntries(TEAMS.map(t=>[t.code,t]));let generated=[],activeStage='Group',teamSort='title',lineupCode=TITLE[0].code;
function flag(code){{let f=byCode[code]?.flag;return f?`<img class="flag" src="https://flagcdn.com/w80/${{f}}.png" alt="${{code}}" onerror="this.replaceWith(fallbackFlag('${{code}}'))">`:`<span class="flagbox">${{code}}</span>`}}
function fallbackFlag(code){{let s=document.createElement('span');s.className='flagbox';s.textContent=code;return s}}
function pct(x){{return `${{(Number(x||0)*100).toFixed(1)}}%`}}function n(x){{return Number(x||0).toFixed(0)}}
function teamLine(code){{let t=byCode[code]||{{name:code}};return `<div class="teamline">${{flag(code)}}<b>${{t.name}}</b></div>`}}
function ring(label,value,color='var(--green)'){{return `<div class="ring-box"><div class="ring" style="--p:${{Math.max(0,Math.min(100,Number(value||0)))}};--c:${{color}}"><b>${{n(value)}}</b></div><div class="ring-label">${{label}}</div></div>`}}
function showPage(id){{document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));document.getElementById(id).classList.add('active');document.querySelectorAll('.nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===id));window.scrollTo({{top:0,behavior:'smooth'}})}}
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>showPage(b.dataset.page));
function toastMsg(text){{toast.textContent=text;toast.classList.add('show');setTimeout(()=>toast.classList.remove('show'),2300)}}
function initHome(){{podium.innerHTML=TITLE.slice(0,3).map((t,i)=>`<div class="winner-card"><div class="rank">${{i+1}}</div>${{flag(t.code)}}<div><h3>${{t.name}}</h3><p class="small">${{i===0?'Projected champion':i===1?'Runner-up lane':'Third strongest title path'}}</p></div><b>${{pct(t.titleProb)}}</b></div>`).join('');titleBars.innerHTML=TITLE.slice(0,12).map(t=>`<div style="margin:13px 0"><div class="match-top"><b>${{t.name}}</b><span>${{pct(t.titleProb)}}</span></div><div class="probbar"><i style="width:${{t.titleProb*100}}%"></i></div></div>`).join('');notes.innerHTML=`<div class="stat-lines"><div class="stat-line"><b>Winner</b><span class="bar"><i style="width:${{TITLE[0].titleProb*100}}%"></i></span><span>${{TITLE[0].name}}</span></div><div class="stat-line"><b>Bracket</b><span class="bar"><i style="width:100%"></i></span><span>Annex C</span></div><div class="stat-line"><b>Teams</b><span class="bar"><i style="width:100%"></i></span><span>48</span></div><div class="stat-line"><b>Runs</b><span class="bar"><i style="width:100%"></i></span><span>10k</span></div></div>`}}
function groupResults(){{return GROUP_MATCHES.map(m=>{{let aScore=m.pick==='home_win'?2:m.pick==='away_win'?1:1,bScore=m.pick==='away_win'?2:m.pick==='home_win'?1:1;return {{stage:'Group',group:m.group,match:m.match,a:m.aCode,b:m.bCode,aScore,bScore,venue:m.venue,date:m.date,aWin:m.aWin,draw:m.draw,bWin:m.bWin}}}})}}
function baseGroups(){{let out={{}};Object.entries(GROUP_TABLES).forEach(([g,rows])=>out[g]=rows.map(r=>r.code));return out}}
function thirdPool(){{let rows=Object.entries(GROUP_TABLES).map(([g,rs])=>({{group:g,code:rs[2].code,points:rs[2].points,gd:rs[2].gd,power:byCode[rs[2].code]?.power||0}}));return rows.sort((a,b)=>b.points-a.points||b.gd-a.gd||b.power-a.power).slice(0,8)}}
function allocateThirds(pool){{let byGroup=Object.fromEntries(pool.map(t=>[t.group,t]));let key=Object.keys(byGroup).sort().join('');let row=ANNEX_C[key];if(!row)throw new Error(`No FIFA Annex C allocation for ${{key}}`);let assigned={{}};THIRD_SLOT_ORDER.forEach(slot=>{{let group=row[slot].replace('3','');assigned[slot]=byGroup[group].code}});return assigned}}
function resolve(slot,groups,thirds,match){{if(slot[0]==='1'||slot[0]==='2')return groups[slot[1]][Number(slot[0])-1];return thirds[match]}}
function winProb(a,b){{let A=byCode[a],B=byCode[b];let edge=(A.power-B.power)*.42+(A.attack-B.defense)*.22+(A.creation-B.creation)*.14+(A.keeper-B.keeper)*.10+(A.depth-B.depth)*.08;return 1/(1+Math.exp(-edge/18))}}
function playMatch(stage,id,a,b){{let p=winProb(a,b),winner=p>=.5?a:b,loser=winner===a?b:a,close=Math.abs(p-.5)<.08;let aw=winner===a?(close?2:3):(close?1:0),bw=winner===b?(close?2:3):(close?1:0);return {{stage,match:id,a,b,aScore:aw,bScore:bw,winner,loser,prob:p}}}}
function generateTournament(){{let groups=baseGroups(),thirds=allocateThirds(thirdPool()),wins={{}},matches=groupResults();R32.forEach(x=>{{let a=resolve(x[1],groups,thirds,x[0]),b=resolve(x[2],groups,thirds,x[0]),m=playMatch('Round of 32',x[0],a,b);wins[x[0]]=m.winner;matches.push(m)}});[['Round of 16',R16],['Quarterfinal',QF],['Semifinal',SF]].forEach(([stage,arr])=>arr.forEach(x=>{{let m=playMatch(stage,x[0],wins[x[1]],wins[x[2]]);wins[x[0]]=m.winner;matches.push(m)}}));let final=playMatch('Final',FINAL[0],wins[FINAL[1]],wins[FINAL[2]]);matches.push(final);let third=playMatch('Third Place','M103',matches.find(m=>m.match==='M101').loser,matches.find(m=>m.match==='M102').loser);matches.push(third);generated=matches;return matches}}
function initStageTabs(){{stageTabs.innerHTML=STAGES.map(s=>`<button class="chip ${{s===activeStage?'active':''}}" onclick="setStage('${{s}}')">${{s}}</button>`).join('')}}
function setStage(stage){{activeStage=stage;if(stage!=='Group'&&!generated.length)generateTournament();initStageTabs();renderMatches()}}
function matchCard(m){{let stage=m.stage==='Group'?`Group ${{m.group}}`:m.stage;return `<button class="match-card" onclick="openTeam('${{m.winner||m.a}}')"><div class="match-top"><span>${{stage}} · ${{m.match}}</span><span>${{m.venue||''}}</span></div><div class="versus"><div>${{teamLine(m.a)}}</div><div class="score">${{m.aScore}} - ${{m.bScore}}</div><div>${{teamLine(m.b)}}</div></div>${{m.aWin?`<div class="mini-probs"><span><i style="width:${{m.aWin*100}}%"></i></span><span><i style="width:${{m.draw*100}}%;background:var(--gold)"></i></span><span><i style="width:${{m.bWin*100}}%;background:var(--blue)"></i></span></div>`:''}}</button>`}}
function renderMatches(){{let source=activeStage==='Group'?groupResults():(generated.length?generated:generateTournament());let rows=source.filter(m=>m.stage===activeStage);if(activeStage==='Final')rows=source.filter(m=>m.stage==='Final'||m.stage==='Third Place');matchOutput.innerHTML=rows.map(matchCard).join('')}}
function sortTeams(kind){{teamSort=kind;renderTeams()}}
function renderTeams(){{let q=teamSearch.value.toLowerCase();let rows=[...TEAMS].filter(t=>t.name.toLowerCase().includes(q)||t.code.toLowerCase().includes(q));let key=teamSort==='attack'?'attack':teamSort==='defense'?'defense':'titleProb';rows.sort((a,b)=>b[key]-a[key]);teamGrid.innerHTML=rows.map(t=>`<button class="team-card" onclick="openTeam('${{t.code}}')"><div class="teamline">${{flag(t.code)}}<b>${{t.name}}</b></div><p class="small">Title ${{pct(t.titleProb)}} · Power ${{t.power.toFixed(1)}}</p><div class="rings">${{ring('ATK',t.attack)}}${{ring('CRE',t.creation,'var(--blue)')}}${{ring('DEF',t.defense,'var(--orange)')}}</div></button>`).join('')}}
function openTeam(code){{let t=byCode[code];teamBrowse.style.display='none';teamDetail.style.display='block';teamDetail.innerHTML=`<button class="ghost" onclick="closeTeam()" style="margin-bottom:14px">Back to all teams</button><div class="detail-layout"><div class="panel detail-card">${{teamHeader(t)}}${{ringsBlock(t)}}${{statsView(t)}}</div><div class="panel"><div class="section-head"><h2>Predicted XI</h2><span class="small">${{LINEUPS[code]?.formation||''}}</span></div>${{pitch(code)}}</div></div>`;showPage('teams')}}
function closeTeam(){{teamDetail.style.display='none';teamBrowse.style.display='block'}}
function teamHeader(t){{return `<div class="detail-title">${{flag(t.code)}}<div><h2>${{t.name}}</h2><p class="small">Title ${{pct(t.titleProb)}} · Final ${{pct(t.finalProb)}} · Power ${{t.power.toFixed(1)}}</p></div></div>`}}
function ringsBlock(t){{return `<div class="rings">${{ring('Attack',t.attack)}}${{ring('Creation',t.creation,'var(--blue)')}}${{ring('Defense',t.defense,'var(--orange)')}}${{ring('Keeper',t.keeper,'var(--red)')}}${{ring('Depth',t.depth)}}${{ring('Trust',t.confidence,'var(--blue)')}}</div>`}}
function statsView(t){{let vals=[['Attack',t.attack],['Chance creation',t.creation],['Defense',t.defense],['Goalkeeper',t.keeper],['Depth',t.depth],['Data confidence',t.confidence]];return `<svg class="radar" viewBox="0 0 360 360">${{radarPolygon(t)}}</svg><div class="stat-lines">${{vals.map(([k,v])=>`<div class="stat-line"><b>${{k}}</b><span class="bar"><i style="width:${{Math.max(0,Math.min(100,v))}}%"></i></span><span>${{n(v)}}</span></div>`).join('')}}</div>`}}
function radarPolygon(t){{let vals=[t.attack,t.creation,t.defense,t.keeper,t.depth,t.confidence],labels=['ATK','CRE','DEF','GK','DEP','TRUST'],cx=180,cy=180,r=126;let axes=labels.map((l,i)=>{{let a=-Math.PI/2+i*Math.PI*2/6,x=cx+Math.cos(a)*148,y=cy+Math.sin(a)*148;return `<line x1="${{cx}}" y1="${{cy}}" x2="${{x}}" y2="${{y}}" stroke="#315249"/><text x="${{x}}" y="${{y}}" fill="#9db6ad" font-size="12" text-anchor="middle">${{l}}</text>`}}).join('');let pts=vals.map((v,i)=>{{let a=-Math.PI/2+i*Math.PI*2/6,rr=r*v/100;return `${{cx+Math.cos(a)*rr}},${{cy+Math.sin(a)*rr}}`}}).join(' ');return `<circle cx="180" cy="180" r="126" fill="none" stroke="#315249"/><circle cx="180" cy="180" r="84" fill="none" stroke="#243d36"/><circle cx="180" cy="180" r="42" fill="none" stroke="#1b302a"/>${{axes}}<polygon points="${{pts}}" fill="rgba(55,212,127,.34)" stroke="#37d47f" stroke-width="3"/>`}}
function roleBase(role){{const map={{GK:[50,91],LB:[18,73],LCB:[38,72],CB:[50,72],RCB:[62,72],RB:[82,73],LWB:[16,61],RWB:[84,61],DM:[50,58],CM:[50,49],LM:[21,47],RM:[79,47],AM:[50,37],LW:[21,24],RW:[79,24],ST:[50,18],CF:[50,18]}};return map[role]||[50,role.includes('B')?70:role.includes('M')?48:22]}}
function spreadValues(center,count,gap=16){{if(count===1)return[center];let start=center-gap*(count-1)/2;return Array.from({{length:count}},(_,i)=>start+i*gap).map(x=>Math.max(14,Math.min(86,x)))}}
function layoutPlayers(players){{let used={{}};let roleGroups={{}};players.forEach(p=>(roleGroups[p.role]??=[]).push(p));let roleX={{LB:18,LWB:16,LCB:35,CB:50,RCB:65,RB:82,RWB:84,DM:50,CM:50,LM:21,RM:79,AM:50,LW:21,RW:79,ST:50,CF:50,GK:50}};let placed=[];Object.entries(roleGroups).forEach(([role,row])=>{{let [baseX,baseY]=roleBase(role);let xs=spreadValues(roleX[role]??baseX,row.length,role==='CB'?18:role==='ST'||role==='CF'?16:14);row.forEach((p,i)=>placed.push({{p,x:xs[i],y:baseY}}))}});placed.sort((a,b)=>a.y-b.y||a.x-b.x);placed.forEach(item=>{{let key=`${{Math.round(item.x/8)}}-${{Math.round(item.y/7)}}`;let seen=used[key]||0;used[key]=seen+1;if(seen){{item.x+=seen%2?5:-5;item.y+=Math.ceil(seen/2)*4}}}});return placed}}
function roleBand(role){{if(role==='GK')return'GK';if(['RB','CB','LB','RWB','LWB','RCB','LCB'].includes(role))return'DF';if(['DM','CM','AM','RM','LM'].includes(role))return'MF';return'FW'}}
function pitch(code){{let data=LINEUPS[code];if(!data)return'<p class="small">No XI available.</p>';let dots=layoutPlayers(data.players).map(o=>{{let initials=o.p.name.split(' ').map(x=>x[0]).join('').slice(0,3);return `<div class="player" style="left:${{o.x}}%;top:${{o.y}}%"><div class="disc">${{initials}}</div><span>${{o.p.name}}</span><small>${{o.p.role}}</small></div>`}}).join('');return `<div class="field"><div class="half"></div><div class="box top"></div><div class="box bottom"></div>${{dots}}</div>`}}
function initLineups(){{let codes=Object.keys(LINEUPS).sort((a,b)=>(byCode[a]?.name||a).localeCompare(byCode[b]?.name||b));lineupList.innerHTML=codes.map(c=>`<button class="${{c===lineupCode?'active':''}}" onclick="selectLineup('${{c}}')">${{teamLine(c)}}<p class="small">${{LINEUPS[c].formation}}</p></button>`).join('');lineupPitch.innerHTML=pitch(lineupCode)}}
function selectLineup(code){{lineupCode=code;initLineups()}}
function predict(){{generateTournament();let final=generated.find(m=>m.stage==='Final');activeStage='Final';initStageTabs();renderMatches();toastMsg(`${{byCode[final.winner].name}} projected champion`);showPage('matches');startMusic()}}
function startMusic(){{bgMusic.volume=.42;bgMusic.play().then(()=>soundBtn.textContent='Mute').catch(()=>soundBtn.textContent='Play')}}
soundBtn.onclick=()=>{{if(bgMusic.paused)startMusic();else{{bgMusic.pause();soundBtn.textContent='Play'}}}};predictBtn.onclick=predict;teamSearch.oninput=renderTeams;
initHome();initStageTabs();renderMatches();renderTeams();initLineups();
</script>
</body>
</html>"""


def main() -> None:
    html = build_html()
    OUT_PATH.write_text(html, encoding="utf-8")
    shutil.copyfile(OUT_PATH, FRESH_OUT_PATH)
    print(f"Built prototype: {OUT_PATH}")
    print(f"Copied prototype: {FRESH_OUT_PATH}")


if __name__ == "__main__":
    main()
