import { Link } from "react-router-dom";

function Navbar() {
  return (
    <div
      style={{
        height: "75px",
        padding: "0 35px",
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        borderBottom: "1px solid #334155",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "14px",
        }}
      >
        <div
          style={{
            width: "48px",
            height: "48px",
            borderRadius: "12px",
            background: "linear-gradient(135deg, #2563eb, #0ea5e9)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            fontSize: "26px",
            boxShadow: "0 4px 12px rgba(37,99,235,0.35)",
          }}
        >
          🛡️
        </div>

        <div>
          <h2
            style={{
              margin: 0,
              color: "white",
              fontSize: "22px",
            }}
          >
            Airport Security AI
          </h2>

          <div
            style={{
              color: "#94a3b8",
              fontSize: "12px",
              letterSpacing: "1px",
            }}
          >
            SURVEILLANCE COMMAND CENTER
          </div>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: "28px",
          alignItems: "center",
        }}
      >
        <Link to="/">Dashboard</Link>
        <Link to="/live">Live Camera</Link>
        <Link to="/upload">Upload</Link>
        <Link to="/alerts">Alerts</Link>
        <Link to="/admin">Admin</Link>
      </div>
    </div>
  );
}

export default Navbar;