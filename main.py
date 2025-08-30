from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.backend.model import predict
from app.backend.database import init_db, fetch_disease
from app.backend.schemas import PredictResponse, DiseaseOut
from app.recommendation.rules import REMEDY_MAP

app = FastAPI(title="Plant Disease Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")
    image_bytes = await file.read()
    disease, confidence = predict(image_bytes)
    remedies = REMEDY_MAP.get(disease, ["Consult local agronomist for tailored advice."])
    return {"disease": disease, "confidence": confidence, "remedies": remedies}

@app.get("/disease/{name}", response_model=DiseaseOut | dict)
def disease_info(name: str):
    row = fetch_disease(name)
    if not row:
        return {"detail": "Not found"}
    return row
