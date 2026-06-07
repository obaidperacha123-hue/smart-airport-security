function Alerts() {
  const alerts = [
    {
      id: 1,
      type: "Unattended Luggage",
      trackId: 4,
      location: "Terminal A",
      time: "10:35 AM",
      status: "High Risk",
    },
    {
      id: 2,
      type: "Unknown Person Detected",
      trackId: 8,
      location: "Gate 4",
      time: "10:42 AM",
      status: "Medium Risk",
    },
  ];

  return (
    <div style={{ padding: "40px" }}>
      <h1>Security Alerts Dashboard</h1>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "20px",
          marginBottom: "25px",
        }}
      >
        <div className="card">
          <h2>2</h2>
          <p>Active Alerts</p>
        </div>

        <div className="card">
          <h2>1</h2>
          <p>High Risk</p>
        </div>

        <div className="card">
          <h2>0</h2>
          <p>Resolved</p>
        </div>
      </div>

      {alerts.map((alert) => (
        <div
          key={alert.id}
          className="card"
          style={{
            marginTop: "20px",
            borderLeft:
              alert.status === "High Risk"
                ? "6px solid #ef4444"
                : "6px solid #f59e0b",
          }}
        >
          <h2>{alert.type}</h2>
          <p>Track ID: {alert.trackId}</p>
          <p>Location: {alert.location}</p>
          <p>Time: {alert.time}</p>
          <p>Status: {alert.status}</p>
        </div>
      ))}
    </div>
  );
}

export default Alerts;