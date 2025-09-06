/*import { useRef, useState } from "react";

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

      // Convert Blob into File for Hugging Face API
      const file = new File([blob], `gesture_${Date.now()}.webm`, { type: "video/webm" });

      try {
        const formData = new FormData();
        formData.append("video_file", file);

        const res = await fetch("https://anudeep05-videomae-api.hf.space/predict", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        console.log("API Result:", data);
        alert("✅ Prediction: " + data.data[0]);

      } catch (err) {
        console.error("Upload failed:", err);
        alert("❌ Upload failed!");
      }

      // Stop camera
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
}*/


import { useRef, useState } from "react";

export default function GestureToText() {
  const [showOptions, setShowOptions] = useState(false);

  // Option 1: Direct redirect to Hugging Face
  const redirectToHuggingFace = () => {
    window.open("https://huggingface.co/spaces/Anudeep05/VideoMAE-API", "_blank");
  };

  // Option 2: Local recording (your original code as backup)
  const videoRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  let chunks = [];

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
    const recorder = new MediaRecorder(stream);

    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);

    recorder.onstop = async () => {
      setLoading(true);
      const blob = new Blob(chunks, { type: "video/webm" });
      const file = new File([blob], `gesture_${Date.now()}.webm`, { type: "video/webm" });
      try {
        const formData = new FormData();
        formData.append("video", file);
        const res = await fetch("http://localhost:5000/upload", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        setPrediction(data.prediction ?? "No prediction");
      } catch (err) {
        setPrediction("Upload failed!");
      }
      setLoading(false);

      // Stop camera
      if (videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
    };

    recorder.start();
    setMediaRecorder(recorder);
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorder) mediaRecorder.stop();
    setRecording(false);
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-3xl font-bold text-center mb-6 text-gray-800">
        Gesture → Text Recognition
      </h2>
      
      {!showOptions ? (
        // Main options screen
        <div className="space-y-4">
          <div className="text-center mb-8">
            <p className="text-lg text-gray-600 mb-4">
              Choose how you want to test the gesture recognition:
            </p>
          </div>

          {/* Option 1: Direct to Hugging Face */}
          <div className="border border-blue-200 rounded-lg p-6 hover:border-blue-400 transition-colors">
            <h3 className="text-xl font-semibold text-blue-800 mb-3">
              🚀 Use Online Interface (Recommended)
            </h3>
            <p className="text-gray-600 mb-4">
              Open the Hugging Face space directly in your browser. 
              You can record or upload videos there and get instant results.
            </p>
            <button
              onClick={redirectToHuggingFace}
              className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-colors"
            >
              🌐 Open Hugging Face Space
            </button>
          </div>

          {/* Option 2: Local interface */}
          <div className="border border-gray-200 rounded-lg p-6 hover:border-gray-400 transition-colors">
            <h3 className="text-xl font-semibold text-gray-800 mb-3">
              🎥 Try Local Interface (Experimental)
            </h3>
            <p className="text-gray-600 mb-4">
              Use the local recording interface. This requires the backend server to be running.
            </p>
            <button
              onClick={() => setShowOptions(true)}
              className="w-full px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-semibold transition-colors"
            >
              📹 Use Local Recording
            </button>
          </div>

          {/* Supported gestures info */}
          <div className="mt-8 p-4 bg-gray-50 rounded-lg">
            <h4 className="font-bold text-gray-700 mb-2">Supported Gestures:</h4>
            <div className="grid grid-cols-2 gap-1 text-sm text-gray-600">
              <span>• AFGHANISTAN</span>
              <span>• AFRICA</span>
              <span>• ANDHRA_PRADESH</span>
              <span>• ARGENTINA</span>
              <span>• DELHI</span>
              <span>• DENMARK</span>
              <span>• ENGLAND</span>
              <span>• GANGTOK</span>
              <span>• GOA</span>
              <span>• GUJARAT</span>
              <span>• HARYANA</span>
              <span>• HIMACHAL_PRADESH</span>
              <span>• JAIPUR</span>
              <span>• JAMMU_AND_KASHMIR</span>
            </div>
          </div>
        </div>
      ) : (
        // Local recording interface (your original code)
        <div>
          <div className="flex items-center mb-4">
            <button
              onClick={() => setShowOptions(false)}
              className="px-3 py-1 text-blue-600 hover:text-blue-800 font-medium"
            >
              ← Back to Options
            </button>
          </div>
          
          <div className="mb-4">
            <video
              ref={videoRef}
              autoPlay
              muted
              className="w-full rounded-lg border-2 border-gray-300"
              style={{ maxHeight: "400px" }}
            />
          </div>
          
          <div className="text-center mb-4">
            {!recording ? (
              <button
                onClick={startRecording}
                disabled={loading}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold"
              >
                {loading ? "🔄 Processing..." : "🎥 Start Recording"}
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-semibold"
              >
                ⏹ Stop Recording
              </button>
            )}
          </div>
          
          {loading && (
            <div className="text-center mb-4">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <p className="mt-2 text-gray-600">⏳ Converting & Uploading...</p>
            </div>
          )}
          
          {prediction && !loading && (
            <div className="p-4 bg-green-100 border border-green-400 text-green-700 rounded-lg">
              <h3 className="text-xl font-bold">Prediction: {prediction}</h3>
            </div>
          )}

          <div className="mt-4 p-4 bg-yellow-100 border border-yellow-400 text-yellow-700 rounded-lg">
            <p className="text-sm">
              <strong>Note:</strong> If this doesn't work, use the "Open Hugging Face Space" option above for guaranteed results.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}