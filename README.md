# Plant Disease Detection — Pre‑Review Prototype

This is a **demo-ready scaffold** you can show in pre-review. It runs end-to-end:
- Upload a leaf image in **Streamlit UI**
- Sends it to a **FastAPI** backend
- Backend uses a **stub predictor** (replace later with your trained CNN/TFLite)
- Returns a **disease label** + **rule-based remedies**
- Optional **SQLite** DB for disease metadata
- **Pytest** unit test for the API
- **Docker** & **docker-compose** to run both UI and API

> ⚠️ For pre-review: this uses a lightweight **stub model** (no training needed).  
> After review, plug in your trained **TensorFlow/TFLite** model by following the notes in `app/backend/model.py`.

---

## Quick Start (Local)

### 1) Create & activate env
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Start the API
```bash
uvicorn app.backend.main:app --reload --port 8000
```
It will auto-create a SQLite database at `app/db/diseases.db` and seed basic data.

### 3) Start the UI (in another terminal)
```bash
streamlit run app/ui/streamlit_app.py
```
The UI defaults to talking to `http://localhost:8000`.

---

## Docker (Optional)

```bash
docker compose up --build
```
- Backend: http://localhost:8000/docs
- UI: http://localhost:8501

---

## Project Structure

```
app/
  backend/
    main.py            # FastAPI endpoints
    model.py           # Stub predictor + plug points for TF/TFLite
    database.py        # SQLite setup + seed
    schemas.py         # Pydantic models
  recommendation/
    rules.py           # Rule-based remedies
  ui/
    streamlit_app.py   # Streamlit UI
  db/
    schema.sql
docker/
  Dockerfile.api
  Dockerfile.ui
  docker-compose.yml
tests/
  test_api.py
requirements.txt
```

---

## Replacing the Stub with Your Trained Model

1. Train your CNN (e.g., MobileNetV2 fine-tuned) and export **one** of:
   - TensorFlow SavedModel directory, or
   - `.h5` Keras model, or
   - **TensorFlow Lite** `.tflite`

2. Put the file/folder under `app/backend/model_artifacts/` and set the env vars:
   - `MODEL_KIND=keras|savedmodel|tflite`
   - `MODEL_PATH=app/backend/model_artifacts/your_model.xxx`

3. The preprocessor expects **RGB 224×224**. Adjust in `model.py` if your model differs.

---

## Postman

Import `postman_collection.json` to test:
- `POST /predict` (multipart image)
- `GET /disease/{name}`

---

## Team Role Mapping (from your doc)

- **Group 1**: Fill `model_artifacts/` + update `model.py` predict()
- **Group 2**: Extend DB fields in `schema.sql`, expand FastAPI endpoints
- **Group 3**: Improve UI, add multilingual support, optimize Docker

Good luck for your pre‑review! ✨
