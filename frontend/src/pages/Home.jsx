import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();

  return (
    <div style={{ padding: "45px" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1.8fr 1fr",
          gap: "28px",
        }}
      >
        <div className="card">
          <p
            style={{
              color: "#38bdf8",
              fontWeight: "bold",
              letterSpacing: "2px",
            }}
          >
            AIRPORT SECURITY COMMAND CENTER
          </p>

          <h1
            style={{
              fontSize: "58px",
              marginTop: "10px",
              marginBottom: "15px",
            }}
          >
            Smart Airport Security
          </h1>

          <p
            style={{
              color: "#cbd5e1",
              fontSize: "20px",
              lineHeight: "1.7",
            }}
          >
            Advanced computer vision dashboard for live CCTV monitoring,
            unattended luggage detection, object tracking, face recognition,
            and real-time airport security alerts.
          </p>

          <div
            style={{
              display: "flex",
              gap: "18px",
              marginTop: "30px",
            }}
          >
            <button onClick={() => navigate("/live")}>
              Launch Live Feed
            </button>

            <button onClick={() => navigate("/upload")}>
              Analyze Footage
            </button>
          </div>
        </div>

        <div className="card">
          <h2>System Status</h2>

          <div style={statusRow}>
            <span>Camera Feed</span>
            <span style={{ color: "#22c55e" }}>ONLINE</span>
          </div>

          <div style={statusRow}>
            <span>Upload Module</span>
            <span style={{ color: "#22c55e" }}>READY</span>
          </div>

          <div style={statusRow}>
            <span>Alert Engine</span>
            <span style={{ color: "#22c55e" }}>READY</span>
          </div>

          <div style={statusRow}>
            <span>Backend AI</span>
            <span style={{ color: "#f59e0b" }}>PENDING</span>
          </div>
        </div>
      </div>

      <div
        style={{
          marginTop: "30px",
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "20px",
        }}
      >
        <div
          className="card actionCard"
          onClick={() => navigate("/live")}
        >
          <h3>Live Camera</h3>
          <p>
            Monitor airport camera feed in real time.
          </p>
        </div>

        <div
          className="card actionCard"
          onClick={() => navigate("/upload")}
        >
          <h3>Video Analysis</h3>
          <p>
            Upload CCTV footage for AI processing.
          </p>
        </div>

        <div
          className="card actionCard"
          onClick={() => navigate("/alerts")}
        >
          <h3>Threat Alerts</h3>
          <p>
            View suspicious activity and luggage alerts.
          </p>
        </div>

        <div
          className="card actionCard"
          onClick={() => navigate("/admin")}
        >
          <h3>Face Enrolment</h3>
          <p>
            Manage authorised personnel and travellers.
          </p>
        </div>
      </div>
    </div>
  );
}

const statusRow = {
  display: "flex",
  justifyContent: "space-between",
  padding: "14px 0",
  borderBottom: "1px solid #334155",
};

export default Home;