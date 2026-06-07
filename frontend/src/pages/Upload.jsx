import { useState } from "react";

const API_URL = "http://localhost:8000";

function Upload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("No file selected");

  function handleFileChange(event) {
    setFile(event.target.files[0]);
    setStatus("File selected and ready for upload");
  }

  async function handleUpload() {
    if (!file) {
      setStatus("Please select an image or video first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setStatus("Uploading to backend...");

      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();
      console.log(data);

      setStatus("Upload successful. Backend processed the file.");
    } catch (error) {
      setStatus("Backend not connected yet. File is ready for future upload.");
    }
  }

  return (
    <div style={{ padding: "40px" }}>
      <div className="card">
        <h1>Upload CCTV Image / Video</h1>

        <p style={{ color: "#94a3b8" }}>
          Upload airport CCTV footage for AI security analysis.
        </p>

        <input type="file" accept="image/*,video/*" onChange={handleFileChange} />

        <br />
        <br />

        {file && (
          <div className="card" style={{ marginTop: "20px" }}>
            <h3>Selected File</h3>
            <p>Name: {file.name}</p>
            <p>Type: {file.type || "Unknown"}</p>
          </div>
        )}

        <br />

        <button onClick={handleUpload}>Upload to System</button>

        <p style={{ marginTop: "20px", color: "#38bdf8" }}>{status}</p>
      </div>
    </div>
  );
}

export default Upload;