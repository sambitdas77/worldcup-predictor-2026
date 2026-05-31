import React, { useMemo, useState } from "react";
import { BarChart3, CalendarDays, Trophy, Users } from "lucide-react";
import { createRoot } from "react-dom/client";
import data from "./data/worldcupData.json";
import "./styles.css";

const byCode = Object.fromEntries(data.teams.map((team) => [team.code, team]));

const ROUND_OF_32 = [
  ["M73", "2A", "2B"],
  ["M76", "1C", "2F"],
  ["M74", "1E", "3ABCDF"],
  ["M75", "1F", "2C"],
  ["M78", "2E", "2I"],
  ["M77", "1I", "3CDFGH"],
  ["M79", "1A", "3CEFHI"],
  ["M80", "1L", "3EHIJK"],
  ["M82", "1G", "3AEHIJ"],
  ["M81", "1D", "3BEFIJ"],
  ["M84", "1H", "2J"],
  ["M83", "2K", "2L"],
  ["M85", "1B", "3EFGIJ"],
  ["M88", "2D", "2G"],
  ["M86", "1J", "2H"],
  ["M87", "1K", "3DEIJL"],
];
const ROUND_OF_16 = [
  ["M90", "M73", "M75"],
  ["M89", "M74", "M77"],
  ["M91", "M76", "M78"],
  ["M92", "M79", "M80"],
  ["M93", "M83", "M84"],
  ["M94", "M81", "M82"],
  ["M95", "M86", "M88"],
  ["M96", "M85", "M87"],
];
const QUARTERS = [["M97", "M89", "M90"], ["M98", "M93", "M94"], ["M99", "M91", "M92"], ["M100", "M95", "M96"]];
const SEMIS = [["M101", "M97", "M98"], ["M102", "M99", "M100"]];
const THIRD_SLOT_ORDER = ["M79", "M85", "M81", "M74", "M82", "M77", "M87", "M80"];
const STAGES = ["Group Stage", "Round of 32", "Round of 16", "Quarter Finals", "Semi Finals", "Third Place", "Final"];

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function flag(code, className = "flag") {
  const team = byCode[code];
  if (!team?.flag) return <span className="flagFallback">{code}</span>;
  return <img className={className} src={`https://flagcdn.com/w80/${team.flag}.png`} alt={code} />;
}

function winProb(aCode, bCode) {
  const a = byCode[aCode];
  const b = byCode[bCode];
  if (!a || !b) return 0.5;
  const edge =
    (a.power - b.power) * 0.42 +
    (a.attack - b.defense) * 0.22 +
    (a.creation - b.creation) * 0.14 +
    (a.keeper - b.keeper) * 0.1 +
    (a.depth - b.depth) * 0.08;
  return 1 / (1 + Math.exp(-edge / 18));
}

function knockoutMatch(stage, match, aCode, bCode) {
  const aWin = winProb(aCode, bCode);
  const bWin = 1 - aWin;
  return {
    stage,
    match,
    aCode,
    bCode,
    aWin,
    draw: 0,
    bWin,
    winner: aWin >= bWin ? aCode : bCode,
  };
}

function buildTournamentMatches() {
  const grouped = data.groupTables || {};
  const groupLetters = Object.keys(grouped).sort();
  const thirdPool = groupLetters
    .filter((group) => grouped[group]?.[2])
    .map((group) => {
      const row = grouped[group][2];
      return { group, code: row.code, points: row.points, gd: row.gd, power: byCode[row.code]?.power || 0 };
    })
    .sort((a, b) => b.points - a.points || b.gd - a.gd || b.power - a.power)
    .slice(0, 8);
  const thirdByGroup = Object.fromEntries(thirdPool.map((team) => [team.group, team.code]));
  const annexKey = Object.keys(thirdByGroup).sort().join("");
  const annexRow = data.annexC?.[annexKey] || {};
  const thirds = {};

  THIRD_SLOT_ORDER.forEach((slot) => {
    const annexGroup = annexRow[slot]?.replace("3", "");
    const eligibleGroups = ROUND_OF_32.find(([match]) => match === slot)?.[2]?.replace("3", "").split("") || [];
    const fallbackGroup = eligibleGroups.find((group) => thirdByGroup[group]) || Object.keys(thirdByGroup)[0];
    thirds[slot] = thirdByGroup[annexGroup] || thirdByGroup[fallbackGroup];
  });

  const resolve = (slot, matchId) => {
    if (slot.startsWith("1") || slot.startsWith("2")) {
      return grouped[slot[1]]?.[Number(slot[0]) - 1]?.code;
    }
    return thirds[matchId];
  };

  const wins = {};
  const rounds = {
    "Group Stage": data.groupMatches.map((match) => ({
      stage: "Group Stage",
      match: `M${match.match}`,
      group: match.group,
      aCode: match.aCode,
      bCode: match.bCode,
      aWin: match.aWin,
      draw: match.draw,
      bWin: match.bWin,
      winner: match.pick === "home_win" ? match.aCode : match.pick === "away_win" ? match.bCode : null,
    })),
    "Round of 32": [],
    "Round of 16": [],
    "Quarter Finals": [],
    "Semi Finals": [],
    "Third Place": [],
    Final: [],
  };

  ROUND_OF_32.forEach(([match, aSlot, bSlot]) => {
    const result = knockoutMatch("Round of 32", match, resolve(aSlot, match), resolve(bSlot, match));
    wins[match] = result.winner;
    rounds["Round of 32"].push(result);
  });
  ROUND_OF_16.forEach(([match, a, b]) => {
    const result = knockoutMatch("Round of 16", match, wins[a], wins[b]);
    wins[match] = result.winner;
    rounds["Round of 16"].push(result);
  });
  QUARTERS.forEach(([match, a, b]) => {
    const result = knockoutMatch("Quarter Finals", match, wins[a], wins[b]);
    wins[match] = result.winner;
    rounds["Quarter Finals"].push(result);
  });
  SEMIS.forEach(([match, a, b]) => {
    const result = knockoutMatch("Semi Finals", match, wins[a], wins[b]);
    wins[match] = result.winner;
    wins[`${match}L`] = result.winner === result.aCode ? result.bCode : result.aCode;
    rounds["Semi Finals"].push(result);
  });
  rounds["Third Place"].push(knockoutMatch("Third Place", "M103", wins.M101L, wins.M102L));
  rounds.Final.push(knockoutMatch("Final", "M104", wins.M101, wins.M102));
  return rounds;
}

function roleBase(role) {
  const map = {
    GK: [50, 91],
    LB: [18, 73],
    LCB: [38, 72],
    CB: [50, 72],
    RCB: [62, 72],
    RB: [82, 73],
    LWB: [16, 61],
    RWB: [84, 61],
    DM: [50, 58],
    CM: [50, 49],
    LM: [21, 47],
    RM: [79, 47],
    AM: [50, 37],
    LW: [21, 24],
    RW: [79, 24],
    ST: [50, 18],
    CF: [50, 18],
  };
  return map[role] || [50, role.includes("B") ? 70 : role.includes("M") ? 48 : 22];
}

function spreadValues(center, count, gap = 16) {
  if (count === 1) return [center];
  const start = center - (gap * (count - 1)) / 2;
  return Array.from({ length: count }, (_, i) => Math.max(14, Math.min(86, start + i * gap)));
}

function layoutPlayers(players) {
  const roleGroups = {};
  players.forEach((player) => {
    roleGroups[player.role] ||= [];
    roleGroups[player.role].push(player);
  });

  const roleX = {
    LB: 18,
    LWB: 16,
    LCB: 35,
    CB: 50,
    RCB: 65,
    RB: 82,
    RWB: 84,
    DM: 50,
    CM: 50,
    LM: 21,
    RM: 79,
    AM: 50,
    LW: 21,
    RW: 79,
    ST: 50,
    CF: 50,
    GK: 50,
  };

  const used = {};
  const placed = [];
  Object.entries(roleGroups).forEach(([role, row]) => {
    const [baseX, baseY] = roleBase(role);
    const xs = spreadValues(roleX[role] ?? baseX, row.length, role === "CB" ? 18 : 14);
    row.forEach((player, index) => placed.push({ player, x: xs[index], y: baseY }));
  });

  placed.sort((a, b) => a.y - b.y || a.x - b.x);
  placed.forEach((item) => {
    const key = `${Math.round(item.x / 8)}-${Math.round(item.y / 7)}`;
    const seen = used[key] || 0;
    used[key] = seen + 1;
    if (seen) {
      item.x += seen % 2 ? 5 : -5;
      item.y += Math.ceil(seen / 2) * 4;
    }
  });
  return placed;
}

function Pitch({ code }) {
  const lineup = data.lineups[code];
  if (!lineup) return <div className="empty">No lineup available.</div>;

  return (
    <div>
      <div className="pitch">
        <div className="half" />
        <div className="box top" />
        <div className="box bottom" />
        {layoutPlayers(lineup.players).map(({ player, x, y }) => {
          const initials = player.name
            .split(" ")
            .map((part) => part[0])
            .join("")
            .slice(0, 3);
          return (
            <div className="pitchPlayer" style={{ left: `${x}%`, top: `${y}%` }} key={`${player.role}-${player.name}`}>
              <div className="disc">{initials}</div>
              <strong>{player.name}</strong>
              <span>{player.role}</span>
            </div>
          );
        })}
      </div>
      <p className="muted">{byCode[code]?.name} - {lineup.formation}</p>
    </div>
  );
}

function StatRing({ label, value }) {
  return (
    <div className="ringWrap">
      <div className="ring" style={{ "--p": Math.max(0, Math.min(100, value)) }}>
        <b>{Math.round(value)}</b>
      </div>
      <span>{label}</span>
    </div>
  );
}

function RadarChart({ team }) {
  const metrics = [
    ["Power", team.power],
    ["Attack", team.attack],
    ["Creation", team.creation],
    ["Defense", team.defense],
    ["Keeper", team.keeper],
    ["Depth", team.depth],
  ];
  const center = 170;
  const radius = 122;
  const points = metrics
    .map(([, value], index) => {
      const angle = -Math.PI / 2 + (index * 2 * Math.PI) / metrics.length;
      const scaled = radius * Math.max(0, Math.min(100, value)) / 100;
      return [center + Math.cos(angle) * scaled, center + Math.sin(angle) * scaled];
    })
    .map(([x, y]) => `${x},${y}`)
    .join(" ");

  return (
    <div className="radarChart">
      <svg viewBox="0 0 340 340" role="img" aria-label={`${team.name} radar chart`}>
        {[0.25, 0.5, 0.75, 1].map((scale) => (
          <polygon
            className="radarGrid"
            key={scale}
            points={metrics
              .map(([,], index) => {
                const angle = -Math.PI / 2 + (index * 2 * Math.PI) / metrics.length;
                return `${center + Math.cos(angle) * radius * scale},${center + Math.sin(angle) * radius * scale}`;
              })
              .join(" ")}
          />
        ))}
        {metrics.map(([label], index) => {
          const angle = -Math.PI / 2 + (index * 2 * Math.PI) / metrics.length;
          return (
            <g key={label}>
              <line x1={center} y1={center} x2={center + Math.cos(angle) * radius} y2={center + Math.sin(angle) * radius} />
              <text x={center + Math.cos(angle) * 148} y={center + Math.sin(angle) * 148}>{label}</text>
            </g>
          );
        })}
        <polygon className="radarFill" points={points} />
        <polygon className="radarStroke" points={points} />
      </svg>
      <div>
        <h3>{team.name} Profile</h3>
        <p>Power, attack, creation, defense, goalkeeper, and depth in one shape.</p>
      </div>
    </div>
  );
}

function MatchCard({ match }) {
  const winner = match.winner ? byCode[match.winner]?.name : "Draw";
  return (
    <article className="matchCard">
      <small>{match.group ? `Group ${match.group} - ` : ""}{match.match}</small>
      <div className="versus">
        <span>{flag(match.aCode)} {byCode[match.aCode]?.name}</span>
        <b>vs</b>
        <span>{flag(match.bCode)} {byCode[match.bCode]?.name}</span>
      </div>
      <p className="winnerLine">Predicted winner: <strong>{winner}</strong></p>
      <div className="odds">
        <span>{byCode[match.aCode]?.name}: <b>{pct(match.aWin)}</b></span>
        {match.draw > 0 && <span>Draw: <b>{pct(match.draw)}</b></span>}
        <span>{byCode[match.bCode]?.name}: <b>{pct(match.bWin)}</b></span>
      </div>
    </article>
  );
}

function App() {
  const [selectedCode, setSelectedCode] = useState(data.teams[0].code);
  const [view, setView] = useState("overview");
  const [matchStage, setMatchStage] = useState("Group Stage");
  const selectedTeam = byCode[selectedCode];

  const contenders = useMemo(() => [...data.teams].sort((a, b) => b.titleProb - a.titleProb), []);
  const tournamentMatches = useMemo(() => buildTournamentMatches(), []);
  const matches = tournamentMatches[matchStage] || [];

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand"><Trophy size={22} /> FIFA WORLCUP 26 PREDICTOR</div>
        <nav>
          {["overview", "matches", "teams", "lineups"].map((item) => (
            <button className={view === item ? "active" : ""} onClick={() => setView(item)} key={item}>
              {item}
            </button>
          ))}
        </nav>
      </header>

      {view === "overview" && (
        <section className="hero">
          <div>
            <p className="eyebrow">Team stats - player-form features - probable XI prediction</p>
            <h1>WHICH NATION TAKES THE ULTIMATE GLORY</h1>
            <p className="lead">Find out which Nation wins the FIFA World Cup 2026, analyzed and predicted using a ML model.</p>
            <div className="heroStats">
              <span><Trophy size={18} /> Winner: {contenders[0].name}</span>
              <span><Users size={18} /> 48 teams</span>
              <span><CalendarDays size={18} /> 72 group matches</span>
              <span><BarChart3 size={18} /> {pct(contenders[0].titleProb)} title chance</span>
            </div>
          </div>
          <div className="podium">
            {contenders.slice(0, 3).map((team, index) => (
              <button className="podiumCard" onClick={() => { setSelectedCode(team.code); setView("teams"); }} key={team.code}>
                <span className="rank">{index + 1}</span>
                {flag(team.code)}
                <strong>{team.name}</strong>
                <b>{pct(team.titleProb)}</b>
              </button>
            ))}
          </div>
        </section>
      )}

      {view === "matches" && (
        <section>
          <div className="sectionHead"><h2>Match Predictions</h2><span>Group stage to final</span></div>
          <div className="stageTabs">
            {STAGES.map((stage) => (
              <button className={matchStage === stage ? "active" : ""} onClick={() => setMatchStage(stage)} key={stage}>
                {stage}
              </button>
            ))}
          </div>
          <div className="matchGrid">
            {matches.map((match) => <MatchCard match={match} key={`${match.stage}-${match.match}`} />)}
          </div>
        </section>
      )}

      {view === "teams" && (
        <section>
          <div className="sectionHead"><h2>Individual Nation Stats</h2><span>Model scores by country</span></div>
          <div className="teamLayout">
            <div className="teamGrid">
              {contenders.map((team) => (
                <button className={team.code === selectedCode ? "teamCard selected" : "teamCard"} onClick={() => setSelectedCode(team.code)} key={team.code}>
                  {flag(team.code)}
                  <strong>{team.name}</strong>
                  <span>{pct(team.titleProb)}</span>
                </button>
              ))}
            </div>
            <aside className="teamPanel">
              <div className="teamTitle">{flag(selectedCode, "bigFlag")}<div><h2>{selectedTeam.name}</h2><p>{pct(selectedTeam.titleProb)} title chance</p></div></div>
              <div className="rings">
                <StatRing label="ATK" value={selectedTeam.attack} />
                <StatRing label="CRE" value={selectedTeam.creation} />
                <StatRing label="DEF" value={selectedTeam.defense} />
                <StatRing label="GK" value={selectedTeam.keeper} />
              </div>
              <RadarChart team={selectedTeam} />
            </aside>
          </div>
        </section>
      )}

      {view === "lineups" && (
        <section>
          <div className="sectionHead"><h2>Probable Playing XI</h2><span>Role-correct pitch view</span></div>
          <div className="lineupPage">
            <div className="teamGrid compact">
              {contenders.map((team) => (
                <button className={team.code === selectedCode ? "teamCard selected" : "teamCard"} onClick={() => setSelectedCode(team.code)} key={team.code}>
                  {flag(team.code)}
                  <strong>{team.name}</strong>
                </button>
              ))}
            </div>
            <Pitch code={selectedCode} />
          </div>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
