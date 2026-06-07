import { Link } from "react-router-dom";

function Navbar() {
  return (
    <div
      style={{
        height: "70px",
        padding: "0 35px",
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        borderBottom: "1px solid #334155",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      <h2 style={{ margin: 0, color: "white" }}>Airport Security AI</h2>

      <div style={{ display: "flex", gap: "28px" }}>
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