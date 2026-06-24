import { useEffect, useState } from "react";

const API_URL = "http://localhost:8000";

const ALERT_LABELS = {
  unattended_bag: "Unattended Luggage",
  owner_left_scene: "Owner Left Scene",
  unauthorized_access: "Unknown Person Detected",
};

const ALERT_COLOURS = {
  unattended_bag: "#ef4444",
  owner_left_scene: "#f59e0b",
  unauthorized_access: "#f97316",
};

function timeAgo(timestamp) {
  const diff = Math.floor(Date.now() / 1000 - timestamp);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  async function fetchAlerts() {
    try {
      const res = await fetch(`${API_URL}/alerts`);
      if (!res.ok) return;
      const data = await res.json();
      // Newest first, deduplicated by alert_id
      const seen = new Set();
      const unique = [...data.alerts].reverse().filter((a) => {
        if (seen.has(a.alert_id)) return false;
        seen.add(a.alert_id);
        return true;
      });
      setAlerts(unique);
    } catch {
      // backend offline — keep showing last known state
    } finally {
      setLoading(false);
    }
  }

  async function clearAlerts() {
    await fetch(`${API_URL}/alerts`, { method: "DELETE" });
    setAlerts([]);
  }

  useEffect(() => {
    fetchAlerts();
    const timer = setInterval(fetchAlerts, 2000);
    return () => clearInterval(timer);
  }, []);

  const highRisk = alerts.filter((a) => a.alert_type === "unattended_bag").length;
  const medRisk = alerts.filter((a) => a.alert_type !== "unattended_bag").length;

  return (
    <div style={{ padding: "40px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "25px" }}>
        <h1>Security Alerts Dashboard</h1>
        <button onClick={clearAlerts} style={{ background: "#334155", padding: "8px 18px" }}>
          Clear All
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px", marginBottom: "25px" }}>
        <div className="card">
          <h2>{alerts.length}</h2>
          <p>Total Alerts</p>
        </div>
        <div className="card">
          <h2 style={{ color: "#ef4444" }}>{highRisk}</h2>
          <p>High Risk</p>
        </div>
        <div className="card">
          <h2 style={{ color: "#f97316" }}>{medRisk}</h2>
          <p>Medium Risk</p>
        </div>
      </div>

      {loading && <p style={{ color: "#94a3b8" }}>Connecting to backend…</p>}

      {!loading && alerts.length === 0 && (
        <div className="card" style={{ textAlign: "center", color: "#94a3b8" }}>
          <p>No alerts detected. Go to Live Camera to start monitoring.</p>
        </div>
      )}

      {alerts.map((alert) => {
        const colour = ALERT_COLOURS[alert.alert_type] || "#94a3b8";
        const label = ALERT_LABELS[alert.alert_type] || alert.alert_type;
        return (
          <div
            key={alert.alert_id}
            className="card"
            style={{ marginTop: "20px", borderLeft: `6px solid ${colour}` }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <h2 style={{ color: colour }}>{label}</h2>
              <span style={{ color: "#94a3b8", fontSize: "14px" }}>{timeAgo(alert.timestamp)}</span>
            </div>
            {alert.track_id != null && <p>Track ID: {alert.track_id}</p>}
            {alert.duration_seconds > 0 && (
              <p>Duration: {alert.duration_seconds.toFixed(0)}s</p>
            )}
            <p style={{ color: "#94a3b8" }}>{alert.message}</p>
          </div>
        );
      })}
    </div>
  );
}

export default Alerts;
