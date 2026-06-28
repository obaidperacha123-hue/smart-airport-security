import { useState } from "react";

const API_URL = "http://localhost:8000";

function Upload() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("No file selected");
  const [results, setResults] = useState(null);

  function handleFileChange(event) {
    setFile(event.target.files[0]);
    setStatus("File selected and ready for upload");
    setResults(null);
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
      setStatus("Upload successful!");
      setResults(data);
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

        {results && (
          <div className="card" style={{ marginTop: "20px" }}>
            <h3>Detection Results</h3>
            <p>Objects Detected: <strong style={{ color: "#38bdf8" }}>{results.summary.unique_tracked_objects}</strong></p>
            <p>Faces Identified: <strong style={{ color: "#38bdf8" }}>{results.summary.unique_persons_identified}</strong></p>
            <p>Frames Processed: <strong style={{ color: "#38bdf8" }}>{results.summary.total_frames_processed}</strong></p>
            <p>Processing Time: <strong style={{ color: "#38bdf8" }}>{results.duration_seconds}s</strong></p>
            <p>Active Alerts: <strong style={{ color: results.alerts.length > 0 ? "#ef4444" : "#22c55e" }}>{results.alerts.length}</strong></p>
            {results.summary.frame_b64 && (
              <div style={{ marginTop: "20px" }}>
                <h3>Processed Frame</h3>
                <img
                  src={`data:image/jpeg;base64,${results.summary.frame_b64}`}
                  style={{ width: "100%", borderRadius: "8px", border: "2px solid #38bdf8" }}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default Upload;