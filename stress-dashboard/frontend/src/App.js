import React, { useEffect, useState } from "react";
import "./App.css";

const BACKEND_URL = "https://stress-prediction-ml-app.onrender.com";

const stressToAngle = (level) => {
  const map = {
    0: -90,
    1: -45,
    2: 0,
    3: 45,
    4: 90,
  };
  return map[level] ?? -90;
};

const stressLabels = {
  0: "Very Low Stress",
  1: "Low Stress",
  2: "Moderate Stress",
  3: "High Stress",
  4: "Very High Stress",
};

const featureNames = [
  "SDRR",
  "RMSSD",
  "KURT",
  "SKEW",
  "MEAN_REL_RR",
  "MEDIAN_REL_RR",
  "SDRR_RMSSD_REL_RR",
  "LF_NU",
  "SAMPEN",
];

function App() {
  const [features, setFeatures] = useState(Array(9).fill(0));
  const [stressCode, setStressCode] = useState(0);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    fetch(`${BACKEND_URL}/`)
      .then((res) => setConnected(res.ok))
      .catch(() => setConnected(false));
  }, []);

  const handleChange = (index, value) => {
    const updated = [...features];
    updated[index] = Number(value);
    setFeatures(updated);
  };

  const predictStress = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      const data = await res.json();
      setStressCode(data.stress_code);
    } catch {
      setConnected(false);
    }
  };

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <img src="/logo.png" alt="StressSense logo" className="app-logo" />
        <h1 className="app-title">StressSense</h1>
      </header>

      {/* BACKEND STATUS */}
      <div className="backend-status">
        <span className={`dot ${connected ? "online" : "offline"}`} />
        {connected ? "Backend Connected" : "Backend Not Reachable"}
      </div>

      {/* MAIN CONTENT */}
      <div className="layout">
        {/* GAUGE */}
        <div className="card gauge-card">
          <h2>Stress Meter</h2>
          <div className="gauge-container">
            <div className="gauge-bg" />
            <div
              className="gauge-needle"
              style={{ transform: `rotate(${stressToAngle(stressCode)}deg)` }}
            />
          </div>
          <h3>{stressLabels[stressCode]}</h3>
          <p className="sub">Predicted Stress Level</p>
        </div>

        {/* INPUTS */}
        <div className="card">
          <h2>Input Features</h2>
          <div className="grid">
            {features.map((val, i) => (
              <div key={i} className="input-group">
                <label>{featureNames[i]}</label>
                <input
                  type="number"
                  value={val}
                  onChange={(e) => handleChange(i, e.target.value)}
                />
              </div>
            ))}
          </div>

          <button className="predict-btn" onClick={predictStress}>
            Predict Stress Level
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
