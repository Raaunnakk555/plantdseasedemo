from pydantic import BaseModel

class PredictResponse(BaseModel):
    disease: str
    confidence: float
    remedies: list[str]

class DiseaseOut(BaseModel):
    name: str
    symptoms: str | None = None
    treatment: str | None = None
