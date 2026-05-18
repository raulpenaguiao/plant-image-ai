import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.classifier import LinearProbeClassifier
from app.model import CLIPModel
from app.retrieval import FAISSRetrieval
from app.utils import load_image_from_bytes

limiter = Limiter(key_func=get_remote_address)

clip_model: CLIPModel
classifier: LinearProbeClassifier
retrieval: FAISSRetrieval


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_model, classifier, retrieval
    print("Loading CLIP model...")
    clip_model = CLIPModel()
    print("Loading linear probe classifier...")
    classifier = LinearProbeClassifier()
    print("Loading FAISS index...")
    retrieval = FAISSRetrieval()
    print("All models loaded. Ready.")
    yield


app = FastAPI(title="Plant Disease Classifier", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/pics", StaticFiles(directory="pics"), name="pics")


@app.get("/pics-list")
def pics_list():
    files = sorted(f for f in os.listdir("pics") if not f.startswith("."))
    return JSONResponse(files)


@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("frontend/index.html")


@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    data = await file.read()
    image = load_image_from_bytes(data)
    embedding = clip_model.encode_image(image)
    result = classifier.predict(embedding)
    result["retrieved_examples"] = retrieval.query(embedding)
    return result
