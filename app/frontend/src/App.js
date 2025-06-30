import React, { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from "react-router-dom";
import "./App.css";

const featureLabels = {
  Time: "Transaction Time (seconds)",
  Amount: "Transaction Amount (USD)",
  V1: "Amount Anomaly",
  V2: "Frequency Anomaly",
  V3: "Location Behavior",
  V4: "Merchant Category Behavior",
  V5: "Device/Channel Behavior",
  V6: "Time of Day Variation",
  V7: "Card Usage Anomaly",
  V8: "Transaction Amount Spread",
  V9: "Velocity Pattern",
  V10: "Account Age Anomaly",
  V11: "Transaction Type Behavior",
  V12: "Location Change Frequency",
  V13: "Merchant Category Variance",
  V14: "Device Change Frequency",
  V15: "Time Since Last Transaction",
  V16: "Transaction Amount Trend",
  V17: "Account Balance Anomaly",
  V18: "Transaction Channel Usage",
  V19: "Cardholder Spending Habit",
  V20: "Geographic Distance Between Txns",
  V21: "Transaction Sequence Pattern",
  V22: "Transaction Amount Deviation",
  V23: "New Merchant Frequency",
  V24: "Fraud Indicator Pattern 1",
  V25: "Fraud Indicator Pattern 2",
  V26: "Transaction Time Gap Variation",
  V27: "Suspicious Behavior Score",
  V28: "Anomaly Score Overall",
};

const defaultForm = Object.keys(featureLabels).reduce((acc, key) => {
  acc[key] = "";
  return acc;
}, {});

function FormPage() {
  const [formData, setFormData] = useState(defaultForm);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const payload = Object.fromEntries(
      Object.entries(formData).map(([k, v]) => [k, parseFloat(v) || 0])
    );

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      navigate("/result", { state: { prediction: data } });
    } catch (err) {
      alert("Error: Could not get prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App container">
      <h1 className="title">FraudRadar Detection</h1>
      <form onSubmit={handleSubmit} className="form-grid">
        {Object.keys(featureLabels).map((key) => (
          <label key={key} className="form-label">
            {featureLabels[key]}
            <input
              type="number"
              name={key}
              step="any"
              required
              placeholder={`Enter ${featureLabels[key]}`}
              value={formData[key]}
              onChange={handleChange}
              className="form-input"
            />
          </label>
        ))}

        <button type="submit" className="btn" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>
    </div>
  );
}

function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const prediction = location.state?.prediction;

  if (!prediction) {
    return (
      <div className="App container">
        <h2>No prediction data found.</h2>
        <button className="btn" onClick={() => navigate("/")}>
          Back to Form
        </button>
      </div>
    );
  }

  const isFraud = prediction.fraud === 1 || prediction.fraud === true;

  return (
    <div className="App container result-page">
      <h1 className="title">Prediction Result</h1>
      <div className={`result-card ${isFraud ? "fraud" : "safe"}`}>
        <div className="emoji" role="img" aria-label={isFraud ? "Fraud" : "No Fraud"}>
          {isFraud ? "🚨 Fraud Detected!" : "✅ No Fraud Detected"}
        </div>
        <p className="probability">
          Probability Score: {(prediction.probability * 100).toFixed(2)}%
        </p>
      </div>

      <button className="btn" onClick={() => navigate("/")}>
        🔄 Predict Again
      </button>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<FormPage />} />
        <Route path="/result" element={<ResultPage />} />
      </Routes>
    </Router>
  );
}
