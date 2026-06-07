import { useEffect, useRef, useState } from "react";

function LiveCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [time, setTime] = useState("");
  const [backendStatus, setBackendStatus] = useState("Checking...");

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString());
    }, 1000);

    startCamera();
    checkBackend();

    const backendTimer = setInterval(() => {
      checkBackend();
    }, 5000);

    return () => {
      clearInterval(timer);
      clearInterval(backendTimer);
      stopCamera();
    };
  }, []);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      alert("Camera permission denied or camera not found");
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  async function checkBackend() {
    try {
      const response = await fetch("http://localhost:8000/health");

      if (response.ok) {
        setBackendStatus("ONLINE");
      } else {
        setBackendStatus("OFFLINE");
      }
    } catch {
      setBackendStatus("OFFLINE");
    }
  }

  return (
    <div style={{ padding: "40px" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "2fr 1fr",
          gap: "25px",
        }}
      >
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <div>
              <h1>Live CCTV Monitoring</h1>
              <p style={{ color: "#94a3b8" }}>
                Real-time airport surveillance camera feed
              </p>
            </div>

            <h3>{time}</h3>
          </div>

          <div style={{ display: "flex", justifyContent: "center" }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              width="760"
              style={{
                marginTop: "20px",
                border: "3px solid #334155",
                borderRadius: "18px",
                transform: "scaleX(-1)",
                boxShadow: "0 10px 25px rgba(0,0,0,0.35)",
              }}
            />
          </div>

          <br />

          <button onClick={stopCamera}>Stop Camera</button>
        </div>

        <div className="card">
          <h2>Live Analysis Panel</h2>

          <div style={statusRow}>
            <span>Camera</span>
            <strong style={{ color: "#22c55e" }}>ACTIVE</strong>
          </div>

          <div style={statusRow}>
            <span>Backend</span>
            <strong
              style={{
                color: backendStatus === "ONLINE" ? "#22c55e" : "#ef4444",
              }}
            >
              {backendStatus}
            </strong>
          </div>

          <div style={statusRow}>
            <span>Detection Engine</span>
            <strong
              style={{
                color: backendStatus === "ONLINE" ? "#22c55e" : "#ef4444",
              }}
            >
              {backendStatus}
            </strong>
          </div>

          <div style={statusRow}>
            <span>Tracking Engine</span>
            <strong
              style={{
                color: backendStatus === "ONLINE" ? "#22c55e" : "#ef4444",
              }}
            >
              {backendStatus}
            </strong>
          </div>

          <div style={statusRow}>
            <span>Face Recognition</span>
            <strong
              style={{
                color: backendStatus === "ONLINE" ? "#22c55e" : "#ef4444",
              }}
            >
              {backendStatus}
            </strong>
          </div>

          <hr style={{ borderColor: "#334155", margin: "25px 0" }} />

          <h3>Current Monitoring Summary</h3>

          <div style={statusRow}>
            <span>Objects Detected</span>
            <strong>0</strong>
          </div>

          <div style={statusRow}>
            <span>Tracked Luggage</span>
            <strong>0</strong>
          </div>

          <div style={statusRow}>
            <span>Recognised Faces</span>
            <strong>0</strong>
          </div>

          <div style={statusRow}>
            <span>Active Alerts</span>
            <strong style={{ color: "#22c55e" }}>0</strong>
          </div>
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

export default LiveCamera;