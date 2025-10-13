# Summit.OS Models Registry

This directory holds model files used by services (Fusion, Intelligence, etc.). You can mount it in docker-compose (already configured) and reference models via env variables.

Conventions
- MODEL_REGISTRY: absolute path inside the container, defaults to /models
- Fusion (vision inference)
  - FUSION_ENABLE_VISION_AI=true
  - FUSION_MODEL_PATH=/models/default_vision.onnx (or .pt/.pth)
  - FUSION_CONF_THRESHOLD=0.6
- Intelligence (risk scoring)
  - INTELLIGENCE_ENABLE_XGB=true
  - INTELLIGENCE_MODEL_PATH=/models/risk_model.json (XGBoost Booster JSON)

Notes
- If a model path is not provided, Fusion will attempt /models/default_vision.onnx; Intelligence will attempt /models/risk_model.json.
- If dependencies (onnxruntime, torch, xgboost) are not installed, the services will gracefully fall back: Fusion uses an OpenCV heuristic; Intelligence uses rule-based risk mapping.
- Weights are not committed to the repo. Place your model files under ./models and rebuild or restart services.
