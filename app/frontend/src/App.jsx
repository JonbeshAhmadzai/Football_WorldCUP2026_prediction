import {
  Activity,
  AlertTriangle,
  BarChart3,
  CalendarClock,
  CheckCircle2,
  Clock3,
  Database,
  GitBranch,
  Gauge,
  Info,
  Loader2,
  Maximize2,
  Minimize2,
  Play,
  RefreshCw,
  Route,
  Shield,
  Sparkles,
  Swords,
  Target,
  TrendingUp,
  Trophy,
  X,
  Zap,
} from "lucide-react";
import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";

const tabs = [
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "live", label: "Live Center", icon: Activity },
  { id: "bracket", label: "Bracket", icon: GitBranch },
  { id: "path", label: "Path", icon: Route },
  { id: "predictor", label: "Predictor", icon: Target },
  { id: "results", label: "Results", icon: CheckCircle2 },
  { id: "model", label: "Model", icon: Database },
];

function tabFromHash() {
  const candidate = window.location.hash.replace("#", "");
  return tabs.some((tab) => tab.id === candidate) ? candidate : "overview";
}

const stageLabels = {
  r32: "R32",
  r16: "R16",
  quarter: "QF",
  semi: "SF",
  final: "Final",
  win: "Win",
};

const TeamCodeContext = createContext({});

function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return `${Number(value).toFixed(digits)}%`;
}

function numberText(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  return Number(value).toLocaleString();
}

function signedPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
  const number = Number(value);
  return `${number > 0 ? "+" : ""}${number.toFixed(1)}%`;
}

function compactDate(value) {
  if (!value) return "TBD";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 16);
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function clampPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
  return Math.max(0, Math.min(100, Number(value)));
}

function confidenceTone(value) {
  const number = clampPercent(value);
  if (number >= 68) return "green";
  if (number >= 54) return "amber";
  return "blue";
}

function stageDisplay(stage) {
  const labels = {
    "round-of-32": "Round of 32",
    "round-of-16": "Round of 16",
    quarterfinals: "Quarter-final",
    semifinals: "Semi-final",
    final: "Final",
    "3rd-place-match": "Third place",
  };
  return labels[stage] || stageLabels[stage] || stage;
}

function isFutureDate(value) {
  if (!value) return true;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return true;
  return date >= new Date();
}

function countryCodeToFlag(code) {
  const normalized = String(code || "").trim().toUpperCase();
  if (!/^[A-Z]{2}$/.test(normalized)) return "";
  return String.fromCodePoint(...[...normalized].map((char) => 127397 + char.charCodeAt(0)));
}

async function apiJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed: ${response.status}`);
  }
  return data;
}

function ShellLoading() {
  return (
    <div className="min-h-screen grid place-items-center bg-slate-950 text-white">
      <div className="flex items-center gap-3 rounded-lg bg-white/10 px-5 py-4 text-sm">
        <Loader2 className="h-5 w-5 animate-spin" />
        Loading prediction center
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, detail, tone = "blue" }) {
  return (
    <div className={`stat-card tone-${tone}`}>
      <div className="stat-icon">
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <p className="stat-label">{label}</p>
        <p className="stat-value">{value}</p>
        {detail && <p className="stat-detail">{detail}</p>}
      </div>
    </div>
  );
}

function Panel({ title, subtitle, children, action, className = "" }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-header">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

function StatusBadge({ children, tone = "blue" }) {
  return <span className={`status-badge badge-${tone}`}>{children}</span>;
}

function ConfidenceGauge({ value, label = "Confidence", detail }) {
  const number = clampPercent(value);
  return (
    <div className="confidence-gauge-card">
      <div className="confidence-gauge" style={{ "--gauge": `${number}%` }}>
        <strong>{pct(number, 0)}</strong>
      </div>
      <div>
        <span>{label}</span>
        {detail && <p>{detail}</p>}
      </div>
    </div>
  );
}

function ProbabilityBar({ label, value, tone = "blue" }) {
  const number = clampPercent(value);
  return (
    <div className={`probability-bar tone-${tone}`}>
      <div>
        <span>{label}</span>
        <strong>{pct(number)}</strong>
      </div>
      <i><b style={{ width: `${Math.max(2, number)}%` }} /></i>
    </div>
  );
}

function OperationalTile({ icon: Icon, label, value, detail, tone = "green" }) {
  return (
    <div className={`ops-tile tone-${tone}`}>
      <Icon className="h-4 w-4" />
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        {detail && <small>{detail}</small>}
      </div>
    </div>
  );
}

function TeamButton({ team, onTeamClick, className = "" }) {
  const teamCodes = useContext(TeamCodeContext);
  if (!team) return null;
  const initials = String(team)
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
  const flag = countryCodeToFlag(teamCodes[team]);
  return (
    <button type="button" className={`team-button ${className}`} onClick={(event) => {
      event.stopPropagation();
      onTeamClick?.(team);
    }}>
      {flag ? (
        <span className="team-flag" role="img" aria-label={`${team} flag`}>{flag}</span>
      ) : (
        <span className="team-badge">{initials}</span>
      )}
      <span className="team-label">{team}</span>
    </button>
  );
}

function MatchLine({ team1, team2, score, onTeamClick }) {
  return (
    <div className="match-line">
      <TeamButton team={team1} onTeamClick={onTeamClick} />
      <strong>{score || "vs"}</strong>
      <TeamButton team={team2} onTeamClick={onTeamClick} />
    </div>
  );
}

function LiveTimeline({ match }) {
  const clockText = match?.display_clock || match?.status || "Live";
  const minute = Number.parseInt(String(clockText).match(/\d+/)?.[0] || "0", 10);
  const progress = Math.max(4, Math.min(100, Number.isNaN(minute) ? 6 : (minute / 90) * 100));
  const [homeScore = "", awayScore = ""] = String(match?.score || "").split("-");
  const events = [
    { minute: "0'", label: "Kickoff" },
    homeScore || awayScore ? { minute: clockText, label: `${match.home_team} ${match.score} ${match.away_team}` } : null,
    { minute: "90'", label: "Full time" },
  ].filter(Boolean);
  return (
    <div className="live-timeline">
      <div className="timeline-track">
        <i style={{ width: `${progress}%` }} />
        {events.map((event, index) => (
          <span key={`${event.minute}-${index}`} style={{ left: index === 0 ? "0%" : index === events.length - 1 ? "100%" : `${progress}%` }} />
        ))}
      </div>
      <div className="timeline-events">
        {events.map((event, index) => (
          <p key={`${event.label}-${index}`}>
            <strong>{event.minute}</strong>
            <span>{event.label}</span>
          </p>
        ))}
      </div>
    </div>
  );
}

function RankingBars({ rows, metric, onTeamClick }) {
  const max = Math.max(...rows.map((row) => Number(row[metric] || 0)), 1);
  return (
    <div className="ranking-list">
      {rows.map((row, index) => {
        const value = Number(row[metric] || 0);
        return (
          <div className="ranking-row" key={`${row.team}-${index}`}>
            <div className="ranking-name">
              <span>{index + 1}</span>
              <TeamButton team={row.team} onTeamClick={onTeamClick} />
            </div>
            <div className="ranking-track">
              <div className="ranking-fill" style={{ width: `${Math.max(3, (value / max) * 100)}%` }} />
            </div>
            <div className="ranking-value">
              {pct(value)}
              {metric === "win_pct" && row.win_delta_pct !== null && row.win_delta_pct !== undefined && (
                <small className={Number(row.win_delta_pct) >= 0 ? "delta-up" : "delta-down"}>
                  {signedPct(row.win_delta_pct)}
                </small>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StageChart({ series }) {
  const grouped = useMemo(() => {
    const map = new Map();
    for (const item of series) {
      if (!map.has(item.team)) map.set(item.team, []);
      map.get(item.team).push(item);
    }
    return [...map.entries()].slice(0, 8);
  }, [series]);
  const stages = ["r32", "r16", "quarter", "semi", "final", "win"];
  const colors = ["#2563eb", "#0f766e", "#d97706", "#e11d48", "#7c3aed", "#0891b2", "#475569", "#65a30d"];

  if (!grouped.length) return <div className="empty-state">No stage data available.</div>;

  return (
    <div className="stage-chart">
      <svg viewBox="0 0 720 320" role="img" aria-label="Stage probabilities">
        {[0, 25, 50, 75, 100].map((tick) => (
          <g key={tick}>
            <line x1="60" x2="690" y1={280 - tick * 2.4} y2={280 - tick * 2.4} className="grid-line" />
            <text x="22" y={285 - tick * 2.4} className="axis-label">
              {tick}%
            </text>
          </g>
        ))}
        {stages.map((stage, index) => (
          <g key={stage}>
            <text x={70 + index * 120} y="306" className="axis-label">
              {stageLabels[stage]}
            </text>
          </g>
        ))}
        {grouped.map(([team, values], teamIndex) => {
          const byStage = Object.fromEntries(values.map((item) => [item.stage, item.pct]));
          const points = stages
            .map((stage, index) => {
              const y = 280 - Number(byStage[stage] || 0) * 2.4;
              return `${70 + index * 120},${y}`;
            })
            .join(" ");
          return <polyline key={team} points={points} fill="none" stroke={colors[teamIndex]} strokeWidth="3" strokeLinecap="round" />;
        })}
      </svg>
      <div className="chart-legend">
        {grouped.map(([team], index) => (
          <span key={team}>
            <i style={{ background: colors[index] }} />
            {team}
          </span>
        ))}
      </div>
    </div>
  );
}

function DashboardCommandBar({ data, autoRefresh, refreshStatus, lastRefreshAt, refreshingLive, onToggleAuto, onRefresh }) {
  const qualityStatus = data?.data_health?.quality_status || "unknown";
  const scoreTop3 = data?.metrics?.score_top3_accuracy;
  const airflowDetail = data?.data_health?.quality_summary
    ? `${numberText(data.data_health.quality_summary.passed)} / ${numberText(data.data_health.quality_summary.checks)} checks`
    : "Quality checks pending";
  return (
    <div className="live-toolbar command-center">
      <div className="live-indicator">
        <span className={autoRefresh ? "pulse-dot" : "quiet-dot"} />
        <strong>{autoRefresh ? "Live ESPN polling on" : "Live ESPN polling off"}</strong>
        <small>{lastRefreshAt ? `Last app refresh ${lastRefreshAt}` : refreshStatus}</small>
      </div>
      <div className="ops-grid">
        <OperationalTile icon={Activity} label="Live" value={numberText(data?.metrics?.live_matches)} detail="in play" tone="amber" />
        <OperationalTile icon={Clock3} label="Snapshot" value={data?.snapshot || "Loading"} detail="ESPN data" tone="blue" />
        <OperationalTile icon={Gauge} label="Model" value={data?.selected_model?.kind_label || "Loading"} detail={data?.selected_model?.run_id} tone="green" />
        <OperationalTile icon={CheckCircle2} label="Airflow ready" value={qualityStatus} detail={airflowDetail} tone={qualityStatus === "pass" ? "green" : "amber"} />
        <OperationalTile icon={Target} label="Score model" value={scoreTop3 == null ? "N/A" : pct(Number(scoreTop3) * 100)} detail="top-3 scoreline" tone="purple" />
      </div>
      <div className="live-actions">
        <button className={autoRefresh ? "toggle-button active" : "toggle-button"} onClick={onToggleAuto}>
          {autoRefresh ? "Auto on" : "Auto off"}
        </button>
        <button className="secondary-button compact-button" onClick={onRefresh} disabled={refreshingLive}>
          {refreshingLive ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          Refresh ESPN now
        </button>
      </div>
    </div>
  );
}

function DataTable({ rows, columns, maxRows = 14 }) {
  if (!rows?.length) return <div className="empty-state">No rows for this selection.</div>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, maxRows).map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column.key}>{column.render ? column.render(row[column.key], row) : row[column.key] ?? ""}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Overview({ data, metric, onTeamClick }) {
  const hasLiveMatches = (data.live_matches || []).length > 0;
  const [activeSection, setActiveSection] = useState(hasLiveMatches ? "live" : "ranking");
  const sections = [
    ...(hasLiveMatches ? [{ id: "live", label: "Live Now" }] : []),
    { id: "ranking", label: "Ranking" },
    { id: "path", label: "Path" },
    { id: "r32", label: "Round of 32" },
    { id: "fixtures", label: "Fixtures" },
  ];

  useEffect(() => {
    if (!sections.some((section) => section.id === activeSection)) {
      setActiveSection(sections[0]?.id || "ranking");
    }
  }, [activeSection, sections]);

  return (
    <div className="overview-layout">
      <div className="overview-section-tabs">
        {sections.map((section) => (
          <button
            key={section.id}
            className={activeSection === section.id ? "active" : ""}
            onClick={() => setActiveSection(section.id)}
          >
            {section.label}
          </button>
        ))}
      </div>

      {activeSection === "live" && (
        <Panel title="Live Now" subtitle="Current in-play match state from the latest ESPN scrape." className="overview-focus-panel">
          <DataTable
            rows={data.live_matches || []}
            maxRows={12}
            columns={[
              { key: "display_clock", label: "Clock" },
              { key: "home_team", label: "Team 1", render: (value) => <TeamButton team={value} onTeamClick={onTeamClick} /> },
              { key: "away_team", label: "Team 2", render: (value) => <TeamButton team={value} onTeamClick={onTeamClick} /> },
              { key: "score", label: "Score" },
              { key: "status", label: "Status" },
              { key: "venue", label: "Venue" },
            ]}
          />
        </Panel>
      )}

      {activeSection === "ranking" && (
      <Panel title="Tournament Probability Ranking" subtitle="Simulation output sorted with your dashboard controls." className="overview-focus-panel">
        <RankingBars rows={data.simulation || []} metric={metric} onTeamClick={onTeamClick} />
      </Panel>
      )}

      {activeSection === "path" && (
      <Panel title="Path to the Trophy" subtitle="How the leading teams survive each knockout checkpoint." className="overview-focus-panel chart-panel">
        <StageChart series={data.stage_series || []} />
      </Panel>
      )}

      {activeSection === "r32" && (
      <Panel title="Round of 32 Outlook" subtitle="Actual winners where known, projected winners where the fixture is still pending." className="overview-focus-panel">
        <DataTable
          rows={data.round_of_32 || []}
          maxRows={32}
          columns={[
            { key: "date", label: "Kickoff", render: compactDate },
            { key: "match", label: "Match" },
            { key: "score_type", label: "Type" },
            { key: "score", label: "Score" },
            { key: "advancing_team", label: "Advancer" },
            { key: "confidence_pct", label: "Confidence", render: pct },
          ]}
        />
      </Panel>
      )}

      {activeSection === "fixtures" && (
      <Panel title="Upcoming Fixture Predictions" subtitle="Model predictions for known teams in the tournament fixture list." className="overview-focus-panel">
        <DataTable
          rows={data.predictions || []}
          maxRows={80}
          columns={[
            { key: "date", label: "Date", render: compactDate },
            { key: "team1", label: "Team 1", render: (value) => <TeamButton team={value} onTeamClick={onTeamClick} /> },
            { key: "team2", label: "Team 2", render: (value) => <TeamButton team={value} onTeamClick={onTeamClick} /> },
            { key: "prediction", label: "Pick" },
            { key: "team1_win_pct", label: "Team 1", render: pct },
            { key: "draw_pct", label: "Draw", render: pct },
            { key: "team2_win_pct", label: "Team 2", render: pct },
          ]}
        />
      </Panel>
      )}
    </div>
  );
}

function Bracket({
  stages,
  liveMatches = [],
  selectedModel,
  focusTeam = "All teams",
  bracketMode = "live",
  onBracketModeChange,
  onTeamClick,
  onMatchSelect,
  fullscreen,
  onToggleFullscreen,
}) {
  if (!stages?.length) return <div className="empty-state">No bracket data available.</div>;
  const byStage = Object.fromEntries(stages.map((stage) => [stage.stage, stage.matches || []]));
  const r32 = byStage["round-of-32"] || [];
  const r16 = byStage["round-of-16"] || [];
  const quarters = byStage.quarterfinals || [];
  const semis = byStage.semifinals || [];
  const final = (byStage.final || [])[0];
  const third = (byStage["3rd-place-match"] || [])[0];
  const allMatches = stages.flatMap((stage) => stage.matches || []);
  const confirmed = allMatches.filter((match) => match.mode === "actual" || match.status?.toLowerCase().includes("full")).length;
  const projected = allMatches.filter((match) => match.mode === "projected").length;
  const simulated = allMatches.filter((match) => match.simulation_note).length;
  const live = liveMatches[0];
  const championHint = final?.winner || "TBD";
  const hasFocus = focusTeam && focusTeam !== "All teams";
  const isSimulationMode = bracketMode === "simulation";

  const matchCard = (match, index, compact = false) => {
    if (!match) return <div className="bracket-spacer" key={`spacer-${index}`} />;
    const [score1 = "", score2 = ""] = (match.score || "").split("-");
    const statusText = match.status || "Scheduled";
    const isLive = match.mode === "live" || (!match.winner && !statusText.toLowerCase().includes("scheduled") && !statusText.toLowerCase().includes("full"));
    const isFocused = hasFocus && [match.home_team, match.away_team, match.winner].includes(focusTeam);
    const focusClass = hasFocus ? (isFocused ? "focus-match" : "dim-match") : "";
    return (
      <div
        className={`knockout-card ${match.mode || "pending"} ${isLive ? "live-card" : ""} ${focusClass} ${compact ? "compact" : ""}`}
        key={`${match.date}-${index}`}
        role="button"
        tabIndex={0}
        onClick={() => onMatchSelect?.(match)}
        onKeyDown={(event) => {
          if (event.key === "Enter") onMatchSelect?.(match);
        }}
      >
        <div className="knockout-date">
          {isLive && <span className="mini-live-dot" />}
          {compact ? statusText : compactDate(match.date)}
        </div>
        <div className={`knockout-team ${match.winner === match.home_team ? "winner" : ""}`}>
          <TeamButton team={match.home_team} onTeamClick={onTeamClick} />
          <strong>{score1}</strong>
        </div>
        <div className={`knockout-team ${match.winner === match.away_team ? "winner" : ""}`}>
          <TeamButton team={match.away_team} onTeamClick={onTeamClick} />
          <strong>{score2}</strong>
        </div>
        {match.home_advance_pct != null && match.away_advance_pct != null && (
          <div className="knockout-probability">
            <span style={{ width: `${clampPercent(match.home_advance_pct)}%` }} />
            <b style={{ width: `${clampPercent(match.away_advance_pct)}%` }} />
          </div>
        )}
        {match.winner && <div className="knockout-winner">Advances: {match.winner}</div>}
        {!match.winner && <div className="knockout-status">{statusText}</div>}
        {match.confidence_pct != null && !compact && (
          <div className="knockout-confidence">
            <small>{pct(match.confidence_pct)} confidence</small>
          </div>
        )}
      </div>
    );
  };

  const column = (label, matches, className = "") => (
    <div className={`knockout-column ${className}`}>
      <h3>{label}</h3>
      <div className="knockout-stack">
        {matches.map((match, index) => matchCard(match, index, matches.length > 4))}
      </div>
    </div>
  );

  return (
    <div className="knockout-board">
      <div className="knockout-titlebar">
        <div>
          <strong>{isSimulationMode ? "FIFA World Cup 2026 Simulation Path" : "FIFA World Cup 2026 Live Results Path"}</strong>
          <span>
            {isSimulationMode
              ? "Simulation resolves future placeholders from actual results, in-play leaders, and model projections."
              : "Live results mode shows ESPN actual, in-play, and scheduled bracket state without filling future placeholders."}
          </span>
        </div>
        <div className="bracket-insights">
          <span><b>{confirmed}</b> confirmed</span>
          {isSimulationMode && <span><b>{projected}</b> projected</span>}
          {isSimulationMode && <span><b>{simulated}</b> simulated links</span>}
          <span><b>{isSimulationMode ? championHint : live ? "Live active" : "ESPN state"}</b> {isSimulationMode ? "champion path" : "view"}</span>
          {hasFocus && <span><b>{focusTeam}</b> highlighted</span>}
          <span><b>{selectedModel || "Model"}</b></span>
          <div className="bracket-mode-toggle" role="group" aria-label="Bracket mode">
            <button
              type="button"
              className={bracketMode === "live" ? "active" : ""}
              onClick={() => onBracketModeChange?.("live")}
            >
              Live results
            </button>
            <button
              type="button"
              className={bracketMode === "simulation" ? "active" : ""}
              onClick={() => onBracketModeChange?.("simulation")}
            >
              Run simulation
            </button>
          </div>
          <button type="button" className="icon-chip" onClick={onToggleFullscreen}>
            {fullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
            {fullscreen ? "Exit" : "Fullscreen"}
          </button>
        </div>
      </div>
      <div className="live-ribbon">
        {live ? (
          <>
            <span className="mini-live-dot" />
            <strong>Live now</strong>
            <span>{live.home_team} {live.score} {live.away_team}</span>
            <span>{live.display_clock || live.status}</span>
          </>
        ) : (
          <>
            <span className="quiet-dot" />
            <strong>No live knockout match</strong>
            <span>The board will update when ESPN returns an in-play fixture.</span>
          </>
        )}
      </div>
      <div className="round-guide">
        <span>Round of 32</span>
        <span>Round of 16</span>
        <span>Quarter-finals</span>
        <span>Semi-finals</span>
        <span>Final</span>
        <span>Semi-finals</span>
        <span>Quarter-finals</span>
        <span>Round of 16</span>
        <span>Round of 32</span>
      </div>
      <div className="knockout-grid">
        {column("R32", r32.slice(0, 8), "left")}
        {column("R16", r16.slice(0, 4), "left mid")}
        {column("QF", quarters.slice(0, 2), "left deep")}
        {column("SF", semis.slice(0, 1), "left semi")}
        <div className="final-column">
          <div className="trophy-block">
            <div className="trophy-glow-ring">
              <img src="/assets/world-cup-trophy-clean.png" className="trophy-image" alt="World Cup trophy" />
            </div>
            <span>World Cup Final</span>
          </div>
          {matchCard(final, 0)}
          {third && (
            <div className="third-place">
              <span>Third place</span>
              {matchCard(third, 1, true)}
            </div>
          )}
        </div>
        {column("SF", semis.slice(1, 2), "right semi")}
        {column("QF", quarters.slice(2, 4), "right deep")}
        {column("R16", r16.slice(4, 8), "right mid")}
        {column("R32", r32.slice(8, 16), "right")}
      </div>
    </div>
  );
}

function TournamentPath({ data, selectedTeam, onTeamClick, onMatchSelect }) {
  const teams = (data.simulation || []).map((row) => row.team).filter(Boolean);
  const defaultTeam = selectedTeam && selectedTeam !== "All teams" ? selectedTeam : data.glance?.favorite?.team || teams[0] || "";
  const [pathTeam, setPathTeam] = useState(defaultTeam);

  useEffect(() => {
    if (selectedTeam && selectedTeam !== "All teams") setPathTeam(selectedTeam);
  }, [selectedTeam]);

  const simulation = (data.simulation || []).find((row) => row.team === pathTeam) || {};
  const stageRows = [
    { key: "r32_pct", label: "Round of 32", detail: "Tournament entry checkpoint" },
    { key: "r16_pct", label: "Round of 16", detail: "First survival target" },
    { key: "quarter_pct", label: "Quarter-final", detail: "Deep run threshold" },
    { key: "semi_pct", label: "Semi-final", detail: "Medal contention" },
    { key: "final_pct", label: "Final", detail: "One match from the trophy" },
    { key: "win_pct", label: "Champion", detail: "Lift the trophy" },
  ];
  const bracketMatches = (data.bracket || [])
    .flatMap((stage) => (stage.matches || []).map((match) => ({ ...match, stage: stage.stage })))
    .filter((match) => [match.home_team, match.away_team, match.winner].includes(pathTeam));
  const futurePredictions = (data.predictions || [])
    .filter((row) => row.team1 === pathTeam || row.team2 === pathTeam)
    .filter((row) => isFutureDate(row.date))
    .slice(0, 5);
  const currentStage = [...stageRows].reverse().find((stage) => Number(simulation[stage.key] || 0) > 0);
  const winDelta = simulation.win_delta_pct;

  return (
    <div className="path-page">
      <section className="path-hero">
        <div>
          <div className="eyebrow"><Route className="h-4 w-4" /> Tournament story mode</div>
          <h2>{pathTeam || "Select a team"} path to the trophy</h2>
          <p>Follow one team through survival probabilities, known results, projected knockout opponents, and upcoming model picks.</p>
          <div className="path-controls">
            <label>
              Team path
              <select value={pathTeam} onChange={(event) => setPathTeam(event.target.value)}>
                {teams.map((team) => <option key={team}>{team}</option>)}
              </select>
            </label>
            <button className="secondary-button compact-button" onClick={() => onTeamClick?.(pathTeam)}>
              <Info className="h-4 w-4" />
              Team profile
            </button>
          </div>
        </div>
        <div className="path-trophy-card">
          <img src="/assets/world-cup-trophy-clean.png" alt="" />
          <span>Champion probability</span>
          <strong>{pct(simulation.win_pct)}</strong>
          <small>{winDelta == null ? "No baseline delta" : `${signedPct(winDelta)} vs original model`}</small>
        </div>
      </section>

      <div className="path-grid">
        <Panel title="Stage Survival Ladder" subtitle={`Current checkpoint: ${currentStage?.label || "No stage probability available"}.`} className="path-ladder-panel">
          <div className="path-ladder">
            {stageRows.map((stage, index) => (
              <div className="path-stage-card" key={stage.key}>
                <div>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <strong>{stage.label}</strong>
                  <small>{stage.detail}</small>
                </div>
                <ProbabilityBar label="Chance" value={simulation[stage.key]} tone={stage.key === "win_pct" ? "amber" : "green"} />
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Known and Projected Route" subtitle="Confirmed matches and projected knockout cards involving this team." className="path-route-panel">
          <div className="path-match-list">
            {bracketMatches.length ? bracketMatches.map((match, index) => (
              <div
                className={`path-match-card ${match.mode || "pending"}`}
                key={`${match.date}-${index}`}
                role="button"
                tabIndex={0}
                onClick={() => onMatchSelect?.(match)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") onMatchSelect?.(match);
                }}
              >
                <span>{stageDisplay(match.stage)} | {compactDate(match.date)}</span>
                <MatchLine team1={match.home_team} team2={match.away_team} score={match.score} onTeamClick={onTeamClick} />
                <small>{match.winner ? `Advances: ${match.winner}` : match.status || "Scheduled"}</small>
                {match.confidence_pct != null && <ProbabilityBar label="Advance confidence" value={match.confidence_pct} tone={confidenceTone(match.confidence_pct)} />}
              </div>
            )) : <div className="empty-state">No bracket route is confirmed for this team yet.</div>}
          </div>
        </Panel>

        <Panel title="Upcoming Model Watch" subtitle="Next fixtures where the selected model has known-team predictions." className="path-watch-panel">
          <div className="watch-stack">
            {futurePredictions.length ? futurePredictions.map((row, index) => (
              <div className="watch-card" key={`${row.date}-${index}`}>
                <span>{compactDate(row.date)}</span>
                <MatchLine team1={row.team1} team2={row.team2} onTeamClick={onTeamClick} />
                <ProbabilityBar label={`Pick: ${row.prediction}`} value={row.confidence_pct} tone={confidenceTone(row.confidence_pct)} />
              </div>
            )) : <div className="empty-state">No upcoming known-team prediction is available for this team.</div>}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function Predictor({ options, modelLabel }) {
  const teams = options?.predictor_teams || (options?.teams || []).filter((team) => team !== "All teams");
  const [homeTeam, setHomeTeam] = useState(teams[0] || "");
  const [awayTeam, setAwayTeam] = useState(teams[1] || "");
  const [country, setCountry] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [backtestSource, setBacktestSource] = useState("selected");
  const [backtest, setBacktest] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);

  useEffect(() => {
    if (!homeTeam && teams[0]) setHomeTeam(teams[0]);
    if (!awayTeam && teams[1]) setAwayTeam(teams[1]);
  }, [teams, homeTeam, awayTeam]);

  async function runPrediction() {
    setLoading(true);
    setError("");
    try {
      setPrediction(
        await apiJson("/api/predict-match", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam, model_label: modelLabel, country }),
        }),
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function runBacktest() {
    setBacktestLoading(true);
    setError("");
    try {
      setBacktest(
        await apiJson("/api/backtest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_label: modelLabel, source: backtestSource, limit: 25 }),
        }),
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setBacktestLoading(false);
    }
  }

  return (
    <div className="content-grid">
      <Panel title="Team vs Team Prediction" subtitle="Pick any two teams and compare the selected model's win, draw, and projected score probabilities.">
        <div className="predictor-form match-form">
          <label>
            Team 1
            <select value={homeTeam} onChange={(event) => setHomeTeam(event.target.value)}>
              {teams.map((team) => <option key={team}>{team}</option>)}
            </select>
          </label>
          <label>
            Team 2
            <select value={awayTeam} onChange={(event) => setAwayTeam(event.target.value)}>
              {teams.map((team) => <option key={team}>{team}</option>)}
            </select>
          </label>
          <button
            className="secondary-button swap-button"
            type="button"
            onClick={() => {
              setHomeTeam(awayTeam);
              setAwayTeam(homeTeam);
            }}
          >
            <Swords className="h-4 w-4" />
            Swap
          </button>
          <label>
            Host country
            <input value={country} onChange={(event) => setCountry(event.target.value)} placeholder="Neutral by default" />
          </label>
          <button className="primary-button" onClick={runPrediction} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Predict match
          </button>
        </div>
        {error && <div className="error-box">{error}</div>}
        {prediction && (
          <div className="prediction-result">
            <div className="prediction-arena">
              <div className="team-compare-card">
                <TeamButton team={prediction.home_team} />
                <strong>{pct(prediction.probabilities.home_win_pct)}</strong>
                <span>90-minute win</span>
                <ProbabilityBar label="Advance" value={prediction.probabilities.home_advance_pct} tone="blue" />
              </div>
              <div className="score-command-card">
                <span>Projected score</span>
                <strong>{prediction.projected_score}</strong>
                <small>{prediction.prediction === "Draw" ? "Draw is the strongest 90-minute outcome" : `${prediction.projected_advancer} projected to advance`}</small>
                <ConfidenceGauge value={Math.max(prediction.probabilities.home_win_pct, prediction.probabilities.away_win_pct, prediction.probabilities.draw_pct)} label="Result confidence" detail="Highest 90-minute probability" />
              </div>
              <div className="team-compare-card right">
                <TeamButton team={prediction.away_team} />
                <strong>{pct(prediction.probabilities.away_win_pct)}</strong>
                <span>90-minute win</span>
                <ProbabilityBar label="Advance" value={prediction.probabilities.away_advance_pct} tone="green" />
              </div>
            </div>
            <div className="quality-strip">
              <StatusBadge tone="green">Neutral comparison</StatusBadge>
              <StatusBadge tone="blue">{prediction.model_type}</StatusBadge>
              {prediction.order_invariant && <StatusBadge tone="amber">Swap-safe</StatusBadge>}
              <span>Score model top-3: {prediction.score_model_top3_pct == null ? "N/A" : pct(Number(prediction.score_model_top3_pct) * 100)}</span>
            </div>
            <div className="prob-grid">
              <StatCard icon={Shield} label={`${prediction.home_team} win`} value={pct(prediction.probabilities.home_win_pct)} detail="90-minute result" tone="blue" />
              <StatCard icon={Activity} label="Draw" value={pct(prediction.probabilities.draw_pct)} detail="90-minute result" tone="amber" />
              <StatCard icon={Shield} label={`${prediction.away_team} win`} value={pct(prediction.probabilities.away_win_pct)} detail="90-minute result" tone="green" />
            </div>
            <div className="explanation-grid">
              <div>
                <h3>Prediction Explanation</h3>
                {(prediction.explanation || []).map((line) => <p key={line}>{line}</p>)}
              </div>
              <div>
                <h3>Feature Snapshot</h3>
                <p><span>ELO diff</span><strong>{prediction.feature_summary?.elo_diff ?? "N/A"}</strong></p>
                <p><span>Recent wins diff</span><strong>{prediction.feature_summary?.recent_win_diff ?? "N/A"}</strong></p>
                <p><span>Goal diff edge</span><strong>{prediction.feature_summary?.recent_goal_diff_diff ?? "N/A"}</strong></p>
                <p><span>Expected goals</span><strong>{prediction.expected_goals?.home ?? "N/A"} - {prediction.expected_goals?.away ?? "N/A"}</strong></p>
              </div>
            </div>
            <div className="scoreline-list">
              <h3>Most likely scorelines</h3>
              {(prediction.top_scorelines || []).map((row) => (
                <span key={row.score}>
                  {row.score}
                  <i style={{ width: `${Math.max(8, clampPercent(row.probability_pct) * 4)}px` }} />
                  <b>{pct(row.probability_pct)}</b>
                </span>
              ))}
            </div>
          </div>
        )}
      </Panel>
      <Panel title="Past Match Simulation Check" subtitle="Replay known matches through a saved final model to see whether predictions match actual outcomes.">
        <div className="predictor-form compact">
          <label>
            Data source
            <select value={backtestSource} onChange={(event) => setBacktestSource(event.target.value)}>
              <option value="selected">Selected model data</option>
              <option value="augmented">Historical + ESPN live</option>
              <option value="live">Live result model only</option>
            </select>
          </label>
          <button className="secondary-button" onClick={runBacktest} disabled={backtestLoading}>
            {backtestLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Run check
          </button>
        </div>
        {backtest && (
          <div className="backtest-block">
            <div className="prob-grid">
              <StatCard icon={Target} label="Retrospective accuracy" value={pct(backtest.accuracy_pct)} detail={`${numberText(backtest.rows)} scored matches`} tone="green" />
              <StatCard icon={BarChart3} label="Evaluation accuracy" value={pct(backtest.eval_accuracy_pct)} detail={backtest.model_type} tone="blue" />
            </div>
            <p className="note">{backtest.note}</p>
            <DataTable
              rows={backtest.sample || []}
              maxRows={10}
              columns={[
                { key: "date", label: "Date" },
                { key: "home_team", label: "Team 1" },
                { key: "away_team", label: "Team 2" },
                { key: "target", label: "Actual" },
                { key: "prediction", label: "Predicted" },
                { key: "confidence_pct", label: "Confidence", render: pct },
                { key: "correct", label: "Hit", render: (value) => (value ? "Yes" : "No") },
              ]}
            />
          </div>
        )}
      </Panel>
    </div>
  );
}

function Results({ rows }) {
  return (
    <Panel title="ESPN Match Results" subtitle="Finished, live, and scheduled fixtures scraped into the local dataset.">
      <DataTable
        rows={rows || []}
        maxRows={104}
        columns={[
          { key: "date", label: "Date", render: compactDate },
          { key: "stage", label: "Stage" },
          { key: "home_team", label: "Team 1" },
          { key: "away_team", label: "Team 2" },
          { key: "score", label: "Score" },
          { key: "status", label: "Status" },
          { key: "venue", label: "Venue" },
          { key: "city", label: "City" },
        ]}
      />
    </Panel>
  );
}

function LiveCenter({ data, onTeamClick }) {
  const liveRows = data.live_matches || [];
  const recent = (data.results || []).filter((row) => row.status && row.score).slice(-12).reverse();
  const upcoming = (data.results || []).filter((row) => !row.score).slice(0, 12);
  return (
    <div className="live-center-grid">
      <Panel title="Live Match Center" subtitle="Current in-play state from the latest ESPN scrape." className="live-hero-panel">
        {liveRows.length ? (
          <div className="live-card-grid">
            {liveRows.map((match, index) => (
              <div className="live-match-card" key={`${match.home_team}-${match.away_team}-${index}`}>
                <StatusBadge tone="green">{match.display_clock || match.status}</StatusBadge>
                <div className="live-scoreline">
                  <TeamButton team={match.home_team} onTeamClick={onTeamClick} />
                  <strong>{match.score}</strong>
                  <TeamButton team={match.away_team} onTeamClick={onTeamClick} />
                </div>
                <p>{match.venue || "Venue TBD"} {match.city ? `| ${match.city}` : ""}</p>
                <LiveTimeline match={match} />
              </div>
            ))}
          </div>
        ) : (
          <div className="live-empty-state">
            <Clock3 className="h-5 w-5" />
            <div>
              <strong>No in-play match in the latest ESPN snapshot.</strong>
              <span>The app keeps polling ESPN and will promote the match here when a live state appears.</span>
            </div>
          </div>
        )}
      </Panel>
      <Panel title="Recently Finished" subtitle="Latest scored matches in the local ESPN dataset.">
        <DataTable
          rows={recent}
          maxRows={30}
          columns={[
            { key: "date", label: "Date", render: compactDate },
            { key: "home_team", label: "Team 1", render: (value) => <TeamButton team={value} onTeamClick={onTeamClick} /> },
            { key: "away_team", label: "Team 2", render: (value) => <TeamButton team={value} onTeamClick={onTeamClick} /> },
            { key: "score", label: "Score" },
            { key: "stage", label: "Stage" },
          ]}
        />
      </Panel>
      <Panel title="Upcoming Watchlist" subtitle="Scheduled fixtures and placeholders still waiting for confirmed teams.">
        <DataTable
          rows={upcoming}
          maxRows={30}
          columns={[
            { key: "date", label: "Date", render: compactDate },
            { key: "home_team", label: "Team 1", render: (value) => <TeamButton team={value} onTeamClick={onTeamClick} /> },
            { key: "away_team", label: "Team 2", render: (value) => <TeamButton team={value} onTeamClick={onTeamClick} /> },
            { key: "stage", label: "Stage" },
            { key: "venue", label: "Venue" },
          ]}
        />
      </Panel>
    </div>
  );
}

function TeamProfileDrawer({ team, data, onClose }) {
  if (!team || team === "All teams") return null;
  const simulation = (data?.simulation || []).find((row) => row.team === team) || {};
  const results = (data?.results || []).filter((row) => row.home_team === team || row.away_team === team).slice(0, 8);
  const predictions = (data?.predictions || []).filter((row) => row.team1 === team || row.team2 === team).slice(0, 8);
  const stages = (data?.stage_series || []).filter((row) => row.team === team);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="side-drawer" onClick={(event) => event.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <span>Team Profile</span>
            <h2>{team}</h2>
          </div>
          <button className="icon-only" onClick={onClose}><X className="h-4 w-4" /></button>
        </div>
        <div className="drawer-stat-grid">
          <StatCard icon={Trophy} label="Win" value={pct(simulation.win_pct)} tone="purple" />
          <StatCard icon={Target} label="Final" value={pct(simulation.final_pct)} tone="blue" />
          <StatCard icon={GitBranch} label="Semi" value={pct(simulation.semi_pct)} tone="green" />
        </div>
        <div className="profile-section">
          <h3>Path Probabilities</h3>
          <div className="mini-stage-list">
            {stages.map((row) => (
              <p key={row.stage}><span>{stageLabels[row.stage] || row.stage}</span><strong>{pct(row.pct)}</strong></p>
            ))}
          </div>
        </div>
        <div className="profile-section">
          <h3>Matches</h3>
          {results.length ? results.map((row, index) => (
            <p className="profile-row" key={index}>
              <span>{compactDate(row.date)}</span>
              <strong>{row.home_team} {row.score || "vs"} {row.away_team}</strong>
            </p>
          )) : <p className="note">No visible matches for this team in the current filter.</p>}
        </div>
        <div className="profile-section">
          <h3>Upcoming Predictions</h3>
          {predictions.length ? predictions.map((row, index) => (
            <p className="profile-row" key={index}>
              <span>{compactDate(row.date)}</span>
              <strong>{row.team1} vs {row.team2}</strong>
              <em>{row.prediction} | {pct(row.confidence_pct)}</em>
            </p>
          )) : <p className="note">No upcoming predictions for this team in the current filter.</p>}
        </div>
      </aside>
    </div>
  );
}

function MatchDetailDrawer({ match, onClose, onTeamClick }) {
  if (!match) return null;
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="side-drawer match-drawer" onClick={(event) => event.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <span>{match.mode || "match"}</span>
            <h2>{match.home_team} vs {match.away_team}</h2>
          </div>
          <button className="icon-only" onClick={onClose}><X className="h-4 w-4" /></button>
        </div>
        <div className="match-detail-score">
          <TeamButton team={match.home_team} onTeamClick={onTeamClick} />
          <strong>{match.score || "TBD"}</strong>
          <TeamButton team={match.away_team} onTeamClick={onTeamClick} />
        </div>
        <div className="quality-strip">
          <StatusBadge tone={match.mode === "actual" ? "green" : match.mode === "projected" ? "blue" : "amber"}>{match.mode || "pending"}</StatusBadge>
          <span>{compactDate(match.date)}</span>
          <span>{match.status || "Scheduled"}</span>
        </div>
        {match.winner && <p className="drawer-callout">Projected/confirmed advancer: <strong>{match.winner}</strong></p>}
        <div className="profile-section">
          <h3>Top Scorelines</h3>
          {(match.top_scorelines || []).length ? (
            <div className="scoreline-list flat">
              {match.top_scorelines.map((row) => <span key={row.score}>{row.score} <b>{pct(row.probability_pct)}</b></span>)}
            </div>
          ) : (
            <p className="note">No scoreline distribution is available for this match yet.</p>
          )}
        </div>
      </aside>
    </div>
  );
}

function ModelComparison({ registry = [] }) {
  if (!registry.length) return <div className="empty-state">No model registry entries available.</div>;
  const maxRows = Math.max(...registry.map((item) => Number(item.training_rows || 0)), 1);
  const maxEval = Math.max(...registry.map((item) => Number(item.eval_accuracy || 0)), 0.01);
  return (
    <div className="model-comparison">
      {registry.map((item) => (
        <div className="model-compare-card" key={item.label}>
          <div>
            <span>{item.kind_label}</span>
            <strong>{item.label}</strong>
            <small>{item.run_id}</small>
          </div>
          <ProbabilityBar label="Eval accuracy" value={Number(item.eval_accuracy || 0) * 100} tone="green" />
          <ProbabilityBar label="Training volume" value={(Number(item.training_rows || 0) / maxRows) * 100} tone="blue" />
          <div className="model-meta-row">
            <span>Rows</span>
            <b>{numberText(item.training_rows)}</b>
          </div>
          <div className="model-meta-row">
            <span>Relative eval</span>
            <b>{pct((Number(item.eval_accuracy || 0) / maxEval) * 100, 0)}</b>
          </div>
        </div>
      ))}
    </div>
  );
}

function ModelPage({ data }) {
  const diagnostics = data.model_diagnostics || {};
  const selected = data.selected_model || {};
  const quality = data.data_health?.quality_summary || {};
  return (
    <div className="content-grid">
      <Panel title="Selected Model" subtitle="The dashboard reads versioned artifacts without overwriting the original cloned model.">
        <div className="model-list">
          <p><span>Run</span><strong>{selected.label}</strong></p>
          <p><span>Type</span><strong>{selected.kind_label}</strong></p>
          <p><span>Model</span><strong>{selected.model_path}</strong></p>
          <p><span>Simulation</span><strong>{selected.simulation_path || "No simulation file"}</strong></p>
          <p><span>Features</span><strong>{selected.features_path}</strong></p>
        </div>
      </Panel>
      <Panel title="Feature Importance" subtitle="Top factors learned by the selected model.">
        <RankingBars rows={(diagnostics.feature_importance || []).map((row) => ({ team: row.feature, importance: row.importance * 100 }))} metric="importance" />
      </Panel>
      <Panel title="Model Comparison" subtitle="Compare the historical plus live model with the live-result-only model at a glance.">
        <ModelComparison registry={data.model_registry || []} />
      </Panel>
      <Panel title="Data Health" subtitle="Current ESPN snapshot and model-ready data counts.">
        <div className="health-grid">
          <StatCard icon={CheckCircle2} label="ESPN rows" value={numberText(data.data_health?.espn_rows)} tone="green" />
          <StatCard icon={CalendarClock} label="Fixture rows" value={numberText(data.data_health?.fixture_rows)} tone="amber" />
          <StatCard icon={Database} label="Feature rows" value={numberText(data.data_health?.feature_rows)} tone="blue" />
        </div>
        <div className="quality-strip">
          <span className={`quality-pill status-${data.data_health?.quality_status || "warn"}`}>
            {data.data_health?.quality_status || "unknown"}
          </span>
          <span>{numberText(quality.passed)} / {numberText(quality.checks)} checks passed</span>
          <span>{numberText(quality.warnings)} warnings</span>
          <span>{numberText(quality.errors)} errors</span>
        </div>
        <p className="note">Airflow DAG: <code>src/scheduler/airflow_augmented_worldcup_dag.py</code></p>
      </Panel>
      <Panel title="Model Registry" subtitle="Current model artifacts used by the API and dashboard.">
        <DataTable
          rows={data.model_registry || []}
          maxRows={6}
          columns={[
            { key: "kind_label", label: "Model" },
            { key: "training_rows", label: "Rows", render: numberText },
            { key: "eval_accuracy", label: "Eval", render: (value) => (value == null ? "N/A" : pct(Number(value) * 100)) },
            { key: "last_trained_at", label: "Trained" },
          ]}
        />
      </Panel>
    </div>
  );
}

function App() {
  const [options, setOptions] = useState(null);
  const [model, setModel] = useState("");
  const [team, setTeam] = useState("All teams");
  const [metric, setMetric] = useState("win_pct");
  const [sort, setSort] = useState("desc");
  const [top, setTop] = useState(12);
  const [activeTab, setActiveTab] = useState(() => tabFromHash());
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshingLive, setRefreshingLive] = useState(false);
  const [refreshStatus, setRefreshStatus] = useState("Waiting for first live refresh");
  const [lastRefreshAt, setLastRefreshAt] = useState("");
  const [drawerTeam, setDrawerTeam] = useState("");
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [bracketFullscreen, setBracketFullscreen] = useState(false);
  const [bracketMode, setBracketMode] = useState("live");
  const refreshInFlight = useRef(false);
  const hasLoadedDashboard = useRef(false);

  useEffect(() => {
    apiJson("/api/options")
      .then((payload) => {
        setOptions(payload);
        setModel(payload.default_model);
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    const handleHashChange = () => setActiveTab(tabFromHash());
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  function selectTab(id) {
    setActiveTab(id);
    window.history.replaceState(null, "", `#${id}`);
  }

  async function loadDashboard({ refreshEspn = false } = {}) {
    if (!model) return;
    const params = new URLSearchParams({
      model,
      team,
      metric,
      sort,
      top: String(top),
      simulate_bracket: String(bracketMode === "simulation"),
    });
    if (!hasLoadedDashboard.current) setLoading(true);

    if (refreshEspn && !refreshInFlight.current) {
      refreshInFlight.current = true;
      setRefreshingLive(true);
      try {
        const refresh = await apiJson("/api/refresh-live", { method: "POST" });
        setRefreshStatus(refresh.last_message || refresh.message || "ESPN refresh complete");
        setLastRefreshAt(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      } catch (err) {
        setRefreshStatus(err.message);
      } finally {
        refreshInFlight.current = false;
        setRefreshingLive(false);
      }
    }

    apiJson(`/api/dashboard?${params}`)
      .then((payload) => {
        setData(payload);
        if (!lastRefreshAt) setRefreshStatus(`Snapshot ${payload.snapshot}`);
        hasLoadedDashboard.current = true;
        setError("");
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    loadDashboard({ refreshEspn: false });
  }, [model, team, metric, sort, top, bracketMode]);

  useEffect(() => {
    if (!autoRefresh || !model) return undefined;
    const timer = window.setInterval(() => {
      loadDashboard({ refreshEspn: true });
    }, 20000);
    return () => window.clearInterval(timer);
  }, [autoRefresh, model, team, metric, sort, top, bracketMode]);

  if (!options || !model) return <ShellLoading />;

  const metricLabel = options.ranking_metrics.find((item) => item.value === metric)?.label || "Ranking";
  const isBracketTab = activeTab === "bracket";
  const isLiveTab = activeTab === "live";
  const isPathTab = activeTab === "path";
  const isFocusedWorkspace = isBracketTab || isLiveTab || isPathTab;
  const appClasses = [
    isBracketTab ? "bracket-app" : "",
    bracketFullscreen ? "fullscreen-bracket" : "",
  ].filter(Boolean).join(" ");

  return (
    <TeamCodeContext.Provider value={options.team_codes || {}}>
    <div className={`app-shell ${appClasses}`}>
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-icon"><Trophy className="h-6 w-6" /></div>
          <div>
            <h1>World Cup 2026</h1>
            <p>Prediction Center</p>
            <div className="creator-mark">
              <span>Made by</span>
              <strong>Jonbesh Ahmadzai</strong>
            </div>
          </div>
        </div>
        <div className="control-stack">
          <label>
            Model run
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {options.models.map((item) => <option key={item.label}>{item.label}</option>)}
            </select>
          </label>
          <label>
            Team focus
            <select value={team} onChange={(event) => setTeam(event.target.value)}>
              {options.teams.map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <label>
            Simulation ranking
            <select value={metric} onChange={(event) => setMetric(event.target.value)}>
              {options.ranking_metrics.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
            </select>
          </label>
          <label>
            Teams in charts
            <input type="range" min="5" max="24" value={top} onChange={(event) => setTop(Number(event.target.value))} />
            <span className="range-value">{top} teams</span>
          </label>
          <div className="segmented">
            <button className={sort === "desc" ? "active" : ""} onClick={() => setSort("desc")}>Most</button>
            <button className={sort === "asc" ? "active" : ""} onClick={() => setSort("asc")}>Least</button>
          </div>
        </div>
        {data && (
          <div className="artifact-card">
            <p>Snapshot</p>
            <strong>{data.snapshot}</strong>
            <span>{data.selected_model.kind_label}</span>
            {team !== "All teams" && (
              <button className="drawer-link" onClick={() => setDrawerTeam(team)}>Open team profile</button>
            )}
          </div>
        )}
      </aside>

      <main className={`${isBracketTab ? "main bracket-focus" : "main"} ${isLiveTab ? "live-focus" : ""} ${isPathTab ? "path-focus" : ""}`}>
        {!isFocusedWorkspace && (
          <section className="hero">
            <div>
              <div className="eyebrow"><Sparkles className="h-4 w-4" /> React + FastAPI analytics app</div>
              <h2>Professional match prediction workspace</h2>
              <p>Compare historical plus live training, live-result-only training, ESPN scraping output, tournament simulations, and team-vs-team predictions.</p>
            </div>
            <div className="hero-panel">
              <img src="/assets/world-cup-trophy-clean.png" className="hero-trophy" alt="" />
              <span>Selected model</span>
              <strong>{data?.selected_model?.kind_label || "Loading"}</strong>
              <p>{data?.selected_model?.label}</p>
            </div>
          </section>
        )}

        {error && <div className="error-box">{error}</div>}

        {data && (
          <>
            {!isFocusedWorkspace && (
              <div className="stat-grid">
                <StatCard icon={CheckCircle2} label="Finished matches" value={numberText(data.metrics.finished_matches)} detail="Scraped from ESPN" tone="green" />
                <StatCard icon={Activity} label="Live matches" value={numberText(data.metrics.live_matches)} detail="Current in-play count" tone="amber" />
                <StatCard icon={CalendarClock} label="Scheduled matches" value={numberText(data.metrics.scheduled_matches)} detail="Upcoming fixtures" tone="blue" />
                <StatCard icon={Trophy} label="Favorite" value={data.glance.favorite.team} detail={pct(data.glance.favorite.value)} tone="purple" />
              </div>
            )}

            <nav className="tabs">
              {tabs.map(({ id, label, icon: Icon }) => (
                <button key={id} className={activeTab === id ? "active" : ""} onClick={() => selectTab(id)}>
                  <Icon className="h-4 w-4" />
                  {label}
                </button>
              ))}
            </nav>

            <DashboardCommandBar
              data={data}
              autoRefresh={autoRefresh}
              refreshStatus={refreshStatus}
              lastRefreshAt={lastRefreshAt}
              refreshingLive={refreshingLive}
              onToggleAuto={() => setAutoRefresh((value) => !value)}
              onRefresh={() => loadDashboard({ refreshEspn: true })}
            />

            <div className="workspace">
              {loading && <div className="loading-strip"><Loader2 className="h-4 w-4 animate-spin" /> Refreshing dashboard data</div>}
              {!loading && activeTab === "overview" && <Overview data={data} metric={metric} metricLabel={metricLabel} onTeamClick={setDrawerTeam} />}
              {!loading && activeTab === "live" && <LiveCenter data={data} onTeamClick={setDrawerTeam} />}
              {!loading && activeTab === "bracket" && (
                <div className="bracket-workspace">
                  <Bracket
                    stages={data.bracket}
                    liveMatches={data.live_matches}
                    selectedModel={data.selected_model?.kind_label}
                    focusTeam={team}
                    bracketMode={bracketMode}
                    onBracketModeChange={setBracketMode}
                    onTeamClick={setDrawerTeam}
                    onMatchSelect={setSelectedMatch}
                    fullscreen={bracketFullscreen}
                    onToggleFullscreen={() => setBracketFullscreen((value) => !value)}
                  />
                </div>
              )}
              {!loading && activeTab === "path" && (
                <TournamentPath
                  data={data}
                  selectedTeam={team}
                  onTeamClick={setDrawerTeam}
                  onMatchSelect={setSelectedMatch}
                />
              )}
              {!loading && activeTab === "predictor" && <Predictor options={options} modelLabel={model} />}
              {!loading && activeTab === "results" && <Results rows={data.results} />}
              {!loading && activeTab === "model" && <ModelPage data={data} />}
            </div>
          </>
        )}
        <TeamProfileDrawer team={drawerTeam} data={data} onClose={() => setDrawerTeam("")} />
        <MatchDetailDrawer match={selectedMatch} onClose={() => setSelectedMatch(null)} onTeamClick={setDrawerTeam} />
      </main>
    </div>
    </TeamCodeContext.Provider>
  );
}

export default App;
