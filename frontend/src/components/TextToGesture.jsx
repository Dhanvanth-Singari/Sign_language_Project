export default function TextToGesture() {
  return (
    <div className="card">
      <h2>Text → Gesture</h2>
      <p>Type text and see how it maps to ISL (future feature).</p>
      <textarea placeholder="Enter text here..." />
      <button className="btn-primary">Convert</button>
    </div>
  );
}
