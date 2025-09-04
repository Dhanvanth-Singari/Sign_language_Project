import { useRef, useState } from "react";

export default function GestureToText() {
  const videoRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  let chunks = [];

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;

    const recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);

    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "video/webm" });

      // 🔥 Upload to backend instead of downloading
      const formData = new FormData();
      formData.append("video", blob, `gesture_${Date.now()}.webm`);

      try {
        const res = await fetch("http://localhost:5000/upload", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        alert("✅ Uploaded: " + data.file.filename);
      } catch (err) {
        console.error("Upload failed:", err);
        alert("❌ Upload failed!");
      }

      // 🔒 Close webcam after stopping
      if (videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }

      chunks = [];
    };

    recorder.start();
    setMediaRecorder(recorder);
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
    }
    setRecording(false);
  };

  return (
    <div className="card">
      <h2>Gesture → Text</h2>
      <video
        ref={videoRef}
        autoPlay
        muted
        style={{ width: "100%", borderRadius: "8px" }}
      ></video>

      <div style={{ marginTop: "1rem" }}>
        {!recording ? (
          <button className="btn-primary" onClick={startRecording}>
            🎥 Start Recording
          </button>
        ) : (
          <button
            className="btn-primary"
            style={{ background: "red" }}
            onClick={stopRecording}
          >
            ⏹ Stop Recording
          </button>
        )}
      </div>
    </div>
  );
}
