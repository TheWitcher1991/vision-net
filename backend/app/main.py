from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .service import InferenceService

app = FastAPI(title="VisionNet Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_service = InferenceService()


class LoadModelRequest(BaseModel):
    model_name: str


@app.get("/models")
async def list_models():
    models = inference_service.list_models()
    return {"models": models}


@app.post("/model/load")
async def load_model(request: LoadModelRequest):
    try:
        inference_service.load_model(request.model_name)
        return {"status": "success", "message": f"Model '{request.model_name}' loaded successfully"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model/status")
async def model_status():
    if inference_service.model is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "classes": inference_service.classes,
        "device": inference_service.device,
    }


@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    if inference_service.model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /model/load first.")

    try:
        image_bytes = await image.read()
        result = inference_service.infer(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
