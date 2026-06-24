import { useEffect, useRef, useState } from "react";

const API_URL = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws";

function LiveCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const socketRef = useRef(null);

  const [time, setTime] = useState("");
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [detections, setDetections] = useState({
    tracks: [],
    faces: [],
    alerts: [],
  });

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString());
    }, 1000);

    startCamera();
    checkBackend();
    connectWebSocket();

    const backendTimer = setInterval(checkBackend, 5000);
    const frameSender = setInterval(sendFrameToBackend, 500);

    return () => {
      clearInterval(timer);
      clearInterval(backendTimer);
      clearInterval(frameSender);
      stopCamera();

      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch {
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
      const response = await fetch(`${API_URL}/health`);

      if (response.ok) {
        setBackendStatus("ONLINE");
      } else {
        setBackendStatus("OFFLINE");
      }
    } catch {
      setBackendStatus("OFFLINE");
    }
  }

  function connectWebSocket() {
    try {
      const socket = new WebSocket(WS_URL);
      socketRef.current = socket;

      socket.onopen = () => {
        console.log("WebSocket connected");
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
          console.log(data.error);
          return;
        }

        setDetections({
          tracks: data.tracks || [],
          faces: data.faces || [],
          alerts: data.alerts || [],
        });

        drawResults(data.tracks || [], data.faces || []);
      };

      socket.onerror = () => {
        console.log("WebSocket error");
      };

      socket.onclose = () => {
        console.log("WebSocket closed");
      };
    } catch {
      console.log("WebSocket not connected");
    }
  }

  function sendFrameToBackend() {
    const video = videoRef.current;
    const socket = socketRef.current;

    if (!video || !socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg", 0.7);
    const frameBase64 = imageData.split(",")[1];

    socket.send(
      JSON.stringify({
        frame_b64: frameBase64,
      })
    );
  }

  function drawResults(tracks, faces) {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video) return;

    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;

    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    tracks.forEach((track) => {
      const box = track.bbox;

      const x = box.x1 * scaleX;
      const y = box.y1 * scaleY;
      const width = (box.x2 - box.x1) * scaleX;
      const height = (box.y2 - box.y1) * scaleY;
      const mx = canvas.width - x - width;

      context.strokeStyle = track.is_alert ? "red" : "#38bdf8";
      context.lineWidth = 3;
      context.strokeRect(mx, y, width, height);

      context.fillStyle = track.is_alert ? "red" : "#38bdf8";
      context.font = "16px Arial";
      context.fillText(
        `${track.class_name} ID: ${track.track_id}`,
        mx,
        y - 8
      );
    });

    faces.forEach((face) => {
      const box = face.bbox;

      const x = box.x1 * scaleX;
      const y = box.y1 * scaleY;
      const width = (box.x2 - box.x1) * scaleX;
      const height = (box.y2 - box.y1) * scaleY;
      const mx = canvas.width - x - width;

      context.strokeStyle = face.is_authorized ? "#22c55e" : "#f97316";
      context.lineWidth = 3;
      context.strokeRect(mx, y, width, height);

      context.fillStyle = face.is_authorized ? "#22c55e" : "#f97316";
      context.font = "16px Arial";
      context.fillText(face.name, mx, y - 8);
    });
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

          <div
            style={{
              display: "flex",
              justifyContent: "center",
              position: "relative",
            }}
          >
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

            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                marginTop: "20px",
                width: "760px",
                height: "auto",
                pointerEvents: "none",
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
            <span>WebSocket</span>
            <strong
              style={{
                color:
                  socketRef.current?.readyState === WebSocket.OPEN
                    ? "#22c55e"
                    : "#ef4444",
              }}
            >
              {socketRef.current?.readyState === WebSocket.OPEN
                ? "CONNECTED"
                : "DISCONNECTED"}
            </strong>
          </div>

          <hr style={{ borderColor: "#334155", margin: "25px 0" }} />

          <h3>Current Monitoring Summary</h3>

          <div style={statusRow}>
            <span>Objects Detected</span>
            <strong>{detections.tracks.length}</strong>
          </div>

          <div style={statusRow}>
            <span>Recognised Faces</span>
            <strong>{detections.faces.length}</strong>
          </div>

          <div style={statusRow}>
            <span>Active Alerts</span>
            <strong
              style={{
                color: detections.alerts.length > 0 ? "#ef4444" : "#22c55e",
              }}
            >
              {detections.alerts.length}
            </strong>
          </div>

          {detections.alerts.length > 0 && (
            <div style={{ marginTop: "20px" }}>
              <h3 style={{ color: "#ef4444" }}>Latest Alert</h3>
              <p>{detections.alerts[0].message}</p>
            </div>
          )}
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