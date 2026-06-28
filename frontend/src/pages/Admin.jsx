import { useState, useEffect } from "react";

const API_URL = "http://localhost:8000";

function Admin() {
  const [name, setName] = useState("");
  const [image, setImage] = useState(null);
  const [status, setStatus] = useState("");
  const [enrolled, setEnrolled] = useState([]);

  useEffect(() => {
    fetchEnrolled();
  }, []);

  async function fetchEnrolled() {
    try {
      const res = await fetch(`${API_URL}/enrol`);
      const data = await res.json();
      setEnrolled(Object.entries(data.persons || {}));
    } catch {
      setEnrolled([]);
    }
  }

  async function handleEnrol() {
    if (!name.trim()) {
      setStatus("Please enter a name.");
      return;
    }
    if (!image) {
      setStatus("Please select a photo.");
      return;
    }
    const formData = new FormData();
    formData.append("name", name.trim());
    formData.append("images", image);
    try {
      setStatus("Enrolling...");
      const res = await fetch(`${API_URL}/enrol`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setStatus(`✅ Successfully enrolled "${data.name}"!`);
        setName("");
        setImage(null);
        fetchEnrolled();
      } else {
        setStatus("❌ Enrolment failed. Make sure the photo shows a clear face.");
      }
    } catch {
      setStatus("❌ Could not connect to backend.");
    }
  }

  async function handleDelete(personName) {
    try {
      const res = await fetch(`${API_URL}/enrol/${personName}`, { method: "DELETE" });
      const data = await res.json();
      if (data.success) {
        setStatus(`🗑 Removed "${personName}" from database.`);
        fetchEnrolled();
      }
    } catch {
      setStatus("❌ Could not delete person.");
    }
  }

  return (
    <div style={{ padding: "40px" }}>

      {/* Enrol Form */}
      <div className="card" style={{ marginBottom: "30px" }}>
        <h1>Face Enrolment</h1>
        <p style={{ color: "#94a3b8" }}>
          Register authorised airport staff into the face recognition database.
        </p>

        <div style={{ marginTop: "20px" }}>
          <label style={{ color: "#94a3b8", fontSize: "14px" }}>Full Name</label>
          <br />
          <input
            type="text"
            placeholder="e.g. John Smith"
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={{
              marginTop: "8px", padding: "10px", width: "100%",
              borderRadius: "8px", border: "1px solid #334155",
              background: "#1e293b", color: "white", fontSize: "14px"
            }}
          />
        </div>

        <div style={{ marginTop: "20px" }}>
          <label style={{ color: "#94a3b8", fontSize: "14px" }}>
            Reference Photo (clear front-facing photo)
          </label>
          <br />
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImage(e.target.files[0])}
            style={{ marginTop: "8px" }}
          />
        </div>

        {image && (
          <div style={{ marginTop: "15px" }}>
            <img
              src={URL.createObjectURL(image)}
              alt="preview"
              style={{ width: "150px", height: "150px", objectFit: "cover", borderRadius: "8px", border: "2px solid #0d9488" }}
            />
          </div>
        )}

        <button onClick={handleEnrol} style={{ marginTop: "20px" }}>
          Enrol Person
        </button>

        {status && (
          <p style={{ marginTop: "15px", color: status.includes("✅") ? "#22c55e" : status.includes("❌") ? "#ef4444" : "#38bdf8" }}>
            {status}
          </p>
        )}
      </div>

      {/* Enrolled Persons List */}
      <div className="card">
        <h2>Enrolled Persons</h2>
        <p style={{ color: "#94a3b8" }}>
          {enrolled.length === 0 ? "No persons enrolled yet." : `${enrolled.length} person(s) in the database.`}
        </p>

        {enrolled.map(([personName, count]) => (
          <div key={personName} style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            padding: "12px 16px", marginTop: "10px",
            background: "#1e293b", borderRadius: "8px", border: "1px solid #334155"
          }}>
            <div>
              <p style={{ color: "white", fontWeight: "bold", margin: 0, textTransform: "capitalize" }}>{personName}</p>
              <p style={{ color: "#94a3b8", fontSize: "12px", margin: 0 }}>{count} embedding(s) stored</p>
            </div>
            <button
              onClick={() => handleDelete(personName)}
              style={{ background: "#ef4444", padding: "6px 14px", fontSize: "12px" }}
            >
              Remove
            </button>
          </div>
        ))}
      </div>

    </div>
  );
}

export default Admin;