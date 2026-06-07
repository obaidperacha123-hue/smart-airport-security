import { useState } from "react";

const API_URL = "http://localhost:8000";

function Admin() {
  const [name, setName] = useState("");
  const [faceImage, setFaceImage] = useState(null);
  const [status, setStatus] = useState("No person enrolled yet");

  async function handleEnrol() {
    if (!name || !faceImage) {
      setStatus("Please enter name and upload face image");
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("images", faceImage);

    try {
      setStatus("Sending enrolment data to backend...");

      const response = await fetch(`${API_URL}/enrol`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Enrolment failed");
      }

      const data = await response.json();
      console.log(data);

      setStatus(`${name} enrolled successfully.`);
    } catch (error) {
      setStatus("Backend not connected yet. Enrolment data is ready.");
    }
  }

  return (
    <div style={{ padding: "40px" }}>
      <div className="card">
        <h1>Admin Face Enrolment</h1>

        <p style={{ color: "#94a3b8" }}>
          Upload reference face images for authorised personnel or travellers.
        </p>

        <input
          type="text"
          placeholder="Enter Person Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />

        <br />
        <br />

        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            setFaceImage(e.target.files[0]);
            setStatus("Face image selected and ready for enrolment");
          }}
        />

        <br />
        <br />

        {faceImage && (
          <div className="card" style={{ marginTop: "20px" }}>
            <h3>Selected Face Image</h3>
            <p>{faceImage.name}</p>
          </div>
        )}

        <br />

        <button onClick={handleEnrol}>Enrol Person</button>

        <p style={{ marginTop: "20px", color: "#38bdf8" }}>{status}</p>
      </div>
    </div>
  );
}

export default Admin;