import { useEffect, useState, useRef } from "react";
import { io } from "socket.io-client";

const SOCKET_URL = "http://localhost:8000";

// FAKE GENRE DATA FOR DEMO

const INITIAL_GENRES = [
  { name: "Deep House", count: 24 },
  { name: "Tech House", count: 21 },
  { name: "Progressive House", count: 18 },
  { name: "EDM Pop", count: 15 },
  { name: "Future Bass", count: 14 },
  { name: "Melodic Techno", count: 12 },
  { name: "Bass House", count: 11 },
  { name: "Tropical House", count: 9 },
  { name: "Club Pop", count: 8 },
  { name: "Drum & Bass", count: 7 },
  { name: "Electro House", count: 6 },
  { name: "Afro House", count: 5 },
];

const INITIAL_CHECKINS = [
  {
    name: "Alex M.",
    genres: ["Deep House", "Tech House", "Melodic Techno"],
    time: 2,
  },
  {
    name: "Sarah K.",
    genres: ["EDM Pop", "Future Bass", "Tropical House"],
    time: 5,
  },
  {
    name: "Mike R.",
    genres: ["Tech House", "Bass House", "Drum & Bass"],
    time: 8,
  },
  {
    name: "Jessica L.",
    genres: ["Progressive House", "Deep House", "Club Pop"],
    time: 12,
  },
];

const FAKE_NAMES = [
  "Jordan P.",
  "Taylor S.",
  "Morgan W.",
  "Casey B.",
  "Riley D.",
  "Quinn H.",
  "Andre S.",
  "Jamie F.",
  "Drew K.",
  "Peyton M.",
  "Skyler R.",
  "Reese T.",
  "Finley C.",
  "Parker J.",
  "Blake W.",
];

function useSimulatedCheckins(enabled) {
  const [genres, setGenres] = useState(INITIAL_GENRES);
  const [checkins, setCheckins] = useState(INITIAL_CHECKINS);
  const [totalCheckins, setTotalCheckins] = useState(47);

  useEffect(() => {
    if (!enabled) return;
    const genreNames = INITIAL_GENRES.map((g) => g.name);

    const interval = setInterval(() => {
      if (Math.random() > 0.6) {
        const newName =
          FAKE_NAMES[Math.floor(Math.random() * FAKE_NAMES.length)];
        const shuffled = [...genreNames].sort(() => Math.random() - 0.5);
        const newGenres = shuffled.slice(0, 3);

        setCheckins((prev) => [
          {
            name: newName,
            genres: newGenres,
            time: 0,
          },
          ...prev.slice(0, 9),
        ]);

        setTotalCheckins((prev) => prev + 1);

        setGenres((prev) => {
          const updated = prev.map((g) => {
            if (newGenres.includes(g.name)) {
              return { ...g, count: g.count + 1 };
            }
            return g;
          });
          return updated.sort((a, b) => b.count - a.count);
        });
      }

      setCheckins((prev) => prev.map((c) => ({ ...c, time: c.time + 0.05 })));
    }, 3000);

    return () => clearInterval(interval);
  }, [enabled]);

  return { genres, checkins, totalCheckins };
}

// LIGHT CONTROL SYSTEM

function getLightSettings(hypeScore, energy) {
  const hype = hypeScore || 0;
  const nrg = energy || 0;

  if (hype > 0.7 || nrg > 10) {
    return {
      mode: "STROBE",
      color: "#ff0044",
      intensity: 100,
      bpm: 140,
      description: "High intensity strobe sync'd to beat",
    };
  } else if (hype > 0.4 || nrg > 5) {
    return {
      mode: "PULSE",
      color: "#ff00ff",
      intensity: 75,
      bpm: 128,
      description: "Rhythmic pulse with color wash",
    };
  } else if (hype > 0.2 || nrg > 2) {
    return {
      mode: "WAVE",
      color: "#00ffff",
      intensity: 50,
      bpm: 120,
      description: "Gentle wave across fixtures",
    };
  } else {
    return {
      mode: "AMBIENT",
      color: "#4400ff",
      intensity: 25,
      bpm: 0,
      description: "Low ambient glow",
    };
  }
}

function LightControlPanel({ hypeScore, energy }) {
  const settings = getLightSettings(hypeScore, energy);

  return (
    <div className="light-panel">
      <div className="light-header">
        <span className="light-title">LIGHT CONTROL</span>
        <span className="light-auto">AUTO</span>
      </div>

      <div className="light-mode-display">
        <div
          className={`light-preview ${settings.mode.toLowerCase()}`}
          style={{
            background: settings.color,
            boxShadow: `0 0 ${settings.intensity / 2}px ${settings.color}`,
          }}
        />
        <div className="light-mode-info">
          <div className="light-mode-name">{settings.mode}</div>
          <div className="light-mode-desc">{settings.description}</div>
        </div>
      </div>

      <div className="light-params">
        <div className="light-param">
          <span className="param-label">Color</span>
          <span
            className="param-color"
            style={{ background: settings.color }}
          />
          <span className="param-value">{settings.color}</span>
        </div>
        <div className="light-param">
          <span className="param-label">Intensity</span>
          <div className="param-bar">
            <div
              className="param-fill"
              style={{
                width: `${settings.intensity}%`,
                background: settings.color,
              }}
            />
          </div>
          <span className="param-value">{settings.intensity}%</span>
        </div>
        <div className="light-param">
          <span className="param-label">BPM Sync</span>
          <span className="param-value">
            {settings.bpm > 0 ? settings.bpm : "—"}
          </span>
        </div>
      </div>

      <div className="light-fixtures">
        <div className="fixtures-label">DMX Fixtures (8ch)</div>
        <div className="fixtures-grid">
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <div
              key={i}
              className={`fixture ${settings.mode.toLowerCase()}`}
              style={{
                background: settings.color,
                opacity: 0.3 + (settings.intensity / 100) * 0.7,
                animationDelay: `${i * 0.1}s`,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// COMPONENTS

function LiveVideoFeed({ frameData, width, height }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!frameData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${frameData}`;
  }, [frameData]);

  return (
    <div className="video-container">
      <div className="video-header">
        <span className="live-badge">● LIVE</span>
        <span className="resolution">
          {width}x{height}
        </span>
      </div>
      <canvas ref={canvasRef} className="video-canvas" />
    </div>
  );
}

function MetricBar({ label, value, max, type }) {
  const pct = Math.min(100, (value / max) * 100);

  // Hype: 0% = red, 100% = green
  // Energy: 0% = blue, 100% = orange/red
  let color, gradient;
  if (type === "hype") {
    const hue = (pct / 100) * 120;
    color = `hsl(${hue}, 100%, 50%)`;
    gradient = `linear-gradient(90deg, hsl(0, 100%, 50%), hsl(60, 100%, 50%), hsl(120, 100%, 50%))`;
  } else {
    const hue = 200 - (pct / 100) * 160;
    color = `hsl(${hue}, 100%, 50%)`;
    gradient = `linear-gradient(90deg, hsl(200, 100%, 50%), hsl(60, 100%, 50%), hsl(20, 100%, 50%))`;
  }

  return (
    <div className="metric-bar">
      <div className="metric-label">{label}</div>
      <div className="metric-track">
        <div
          className="metric-fill"
          style={{
            width: `${pct}%`,
            background: gradient,
            clipPath: `inset(0 ${100 - pct}% 0 0)`,
          }}
        />
        <div
          className="metric-glow"
          style={{
            left: `${pct}%`,
            background: color,
            boxShadow: `0 0 15px ${color}, 0 0 30px ${color}`,
          }}
        />
      </div>
      <div className="metric-value" style={{ color }}>
        {type === "hype" ? `${pct.toFixed(0)}%` : value.toFixed(1)}
      </div>
    </div>
  );
}

function Heatmap({ data, zones }) {
  if (!data || data.length === 0) return null;
  const maxVal = Math.max(...data.flat(), 1);

  return (
    <div className="heatmap">
      <div className="heatmap-title">ENERGY ZONES</div>
      <div
        className="heatmap-grid"
        style={{ gridTemplateColumns: `repeat(${data[0]?.length || 8}, 1fr)` }}
      >
        {data.map((row, i) =>
          row.map((val, j) => {
            const intensity = val / maxVal;
            const zone = zones?.[i]?.[j] || "low";
            const colors = { low: 200, medium: 45, high: 0 };
            const h = colors[zone];
            return (
              <div
                key={`${i}-${j}`}
                className="heatmap-cell"
                style={{
                  backgroundColor: `hsla(${h}, 100%, ${30 + intensity * 40}%, ${
                    0.3 + intensity * 0.7
                  })`,
                  boxShadow:
                    intensity > 0.6
                      ? `0 0 ${intensity * 15}px hsla(${h}, 100%, 50%, 0.8)`
                      : "none",
                }}
              />
            );
          })
        )}
      </div>
    </div>
  );
}

function GenreRankings({ genres, checkins, totalCheckins }) {
  const maxCount = genres[0]?.count || 1;

  return (
    <div className="genre-panel">
      <div className="genre-header">
        <div className="genre-title">CROWD MUSIC TASTE</div>
        <div className="checkin-count">{totalCheckins} guests</div>
      </div>

      <div className="genre-list">
        {genres.slice(0, 6).map((genre, i) => (
          <div
            key={genre.name}
            className={`genre-item ${i < 3 ? "top-three" : ""}`}
          >
            <span className={`genre-rank ${i < 3 ? "highlight" : ""}`}>
              #{i + 1}
            </span>
            <span className="genre-name">{genre.name}</span>
            <div className="genre-bar-track">
              <div
                className="genre-bar-fill"
                style={{ width: `${(genre.count / maxCount) * 100}%` }}
              />
            </div>
            <span className="genre-count">{genre.count}</span>
          </div>
        ))}
      </div>

      <div className="recent-checkins">
        <div className="checkins-title">Recent Check-ins</div>
        {checkins.slice(0, 3).map((c, i) => (
          <div key={i} className="checkin-item">
            <span className="checkin-name">{c.name}</span>
            <span className="checkin-time">
              {c.time < 1 ? "just now" : `${Math.floor(c.time)}m ago`}
            </span>
            <div className="checkin-genres">
              {c.genres.slice(0, 2).map((g) => (
                <span key={g} className="genre-tag">
                  {g}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Main App

export default function App() {
  const [connected, setConnected] = useState(false);
  const [data, setData] = useState(null);
  const { genres, checkins, totalCheckins } = useSimulatedCheckins(true);

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ["websocket", "polling"],
      reconnection: true,
    });

    socket.on("connect", () => {
      setConnected(true);
    });
    socket.on("disconnect", () => {
      setConnected(false);
    });
    socket.on("vibe_update", (payload) => {
      setData(payload);
    });

    return () => socket.disconnect();
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1 className="logo">VIBE-CHECK</h1>
        <div className={`status ${connected ? "online" : "offline"}`}>
          <span className="status-dot" />
          {connected ? "LIVE" : "CAMERA OFFLINE"}
          {data && (
            <span className="latency">
              {data.latencyMs}ms | {data.fps} FPS
            </span>
          )}
        </div>
      </header>

      <main className="main">
        {/* TOP SECTION (~1/3): Video + Bars */}
        <div className="top-section">
          <div className="video-wrapper">
            {data ? (
              <LiveVideoFeed
                frameData={data.frame}
                width={data.frameWidth}
                height={data.frameHeight}
              />
            ) : (
              <div className="video-placeholder">
                <div className="spinner" />
                <p>Waiting for camera...</p>
              </div>
            )}
          </div>

          <div className="bars-wrapper">
            <MetricBar
              label="HYPE"
              value={data?.hypeScore || 0}
              max={1}
              type="hype"
            />
            <MetricBar
              label="ENERGY"
              value={data?.meanEnergy || 0}
              max={15}
              type="energy"
            />
          </div>
        </div>

        {/* BOTTOM SECTION (~2/3): Heatmap + Genres + Lights */}
        <div className="bottom-section">
          <div className="heatmap-wrapper">
            {data ? (
              <Heatmap data={data.heatmap} zones={data.zones} />
            ) : (
              <div className="heatmap-placeholder">
                <div className="heatmap-title">ENERGY ZONES</div>
                <p>Waiting for data...</p>
              </div>
            )}
          </div>

          <GenreRankings
            genres={genres}
            checkins={checkins}
            totalCheckins={totalCheckins}
          />

          <LightControlPanel
            hypeScore={data?.hypeScore || 0}
            energy={data?.meanEnergy || 0}
          />
        </div>
      </main>
    </div>
  );
}
