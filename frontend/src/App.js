import React, { useMemo, useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:5000/predict";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const canPredict = useMemo(() => !!file && !loading, [file, loading]);

  const onChoose = (e) => {
    const f = e.target.files?.[0];
    setResult(null);
    setError("");

    if (!f) {
      setFile(null);
      setPreview("");
      return;
    }

    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const onPredict = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("file", file);

      const res = await axios.post(API_URL, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(res.data);
    } catch (err) {
      setError(
        err?.response?.data?.error ||
          err?.message ||
          "Prediction failed. Is Flask running on port 5000?"
      );
    } finally {
      setLoading(false);
    }
  };

  const badgeClass =
    result?.label === "COVID" ? "badge badge-covid" : "badge badge-normal";

  return (
    <div className="page">
      <div className="card">
        <h1>COVID-19 X-ray Classifier</h1>
        <p className="muted">Upload a chest X-ray image to get a prediction.</p>

        <div className="row">
          <input type="file" accept="image/*" onChange={onChoose} />
          <button disabled={!canPredict} onClick={onPredict}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </div>

        {preview && (
          <div className="preview">
            <img src={preview} alt="X-ray preview" />
          </div>
        )}

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <div className={badgeClass}>{result.label}</div>

            <div className="grid">
              <div className="box">
                <div className="label">COVID probability</div>
                <div className="value">{result.prob_covid}</div>
              </div>
              <div className="box">
                <div className="label">Normal probability</div>
                <div className="value">{result.prob_normal}</div>
              </div>
              <div className="box">
                <div className="label">Threshold</div>
                <div className="value">{result.threshold}</div>
              </div>
            </div>

            <p className="muted small">
              Note: This is a demo and not a medical diagnostic tool.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
