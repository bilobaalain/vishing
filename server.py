"""
Vishing Detection Backend Server v5.0 - HYBRID (100% LOCAL)
Combine le modèle TFLite + l'analyse lexicale bilingue pour de meilleurs résultats
"""

import os
import re
import json
import unicodedata
import tempfile
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from collections import Counter

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np

# Base path compatible local + Railway/Docker
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv()

app = FastAPI(title="Vishing Detection API - HYBRID", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metadata = None
tfidf_data = None
scaler_data = None
tflite_interpreter = None
whisper_model = None

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

URGENCY_WORDS = [
    "urgent", "immediately", "right now", "quickly", "without delay",
    "act now", "time sensitive", "as soon as possible",
    "urgence", "immediatement", "tout de suite", "rapidement",
    "sans delai", "agissez maintenant", "au plus vite"
]

SECURITY_WORDS = [
    "security", "secure", "suspicious", "fraud", "fraudulent",
    "unusual activity", "verification", "verify", "blocked",
    "restricted", "account", "safety alert",
    "securite", "suspect", "fraude", "frauduleux",
    "activite inhabituelle", "verification", "verifier",
    "bloque", "restreint", "compte", "alerte de securite"
]

PRESSURE_WORDS = [
    "must", "need to", "have to", "required", "stay on the line",
    "do not hang up", "follow instructions", "now", "mandatory",
    "you need to act", "you must",
    "devez", "il faut", "obligatoire", "restez en ligne",
    "ne raccrochez pas", "suivez les instructions", "maintenant",
    "vous devez agir", "c est obligatoire"
]

SENSITIVE_WORDS = [
    "otp", "code", "password", "pin", "card number", "bank details",
    "identity", "credentials", "login", "access", "security code",
    "verification code", "one time password",
    "mot de passe", "code de verification", "code secret",
    "numero de carte", "coordonnees bancaires", "identite",
    "identifiant", "acces", "code bancaire", "pin bancaire"
]

AUTHORITY_WORDS = [
    "bank", "security department", "customer support", "official",
    "department", "service", "team", "technical support",
    "police", "government", "tax office", "amazon", "paypal",
    "visa", "mastercard",
    "banque", "service securite", "support client", "officiel",
    "departement", "service", "equipe", "support technique",
    "police", "gouvernement", "impots", "amazon", "paypal",
    "visa", "mastercard"
]


def load_whisper_model():
    global whisper_model

    try:
        from faster_whisper import WhisperModel

        print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        print(f"Whisper model loaded: {WHISPER_MODEL_SIZE}")
        return True

    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return False


def load_tflite_model():
    global metadata, tfidf_data, scaler_data, tflite_interpreter

    try:
        metadata_path = BASE_DIR / "metadata.json"
        tfidf_path = BASE_DIR / "tfidf_pruned.json"
        scaler_path = BASE_DIR / "scaler_pruned.json"
        model_path = BASE_DIR / "vishing_dense_pruned.tflite"

        print(f"Loading metadata from: {metadata_path}")
        print(f"Loading TF-IDF from: {tfidf_path}")
        print(f"Loading scaler from: {scaler_path}")
        print(f"Loading TFLite model from: {model_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        with open(tfidf_path, "r", encoding="utf-8") as f:
            tfidf_data = json.load(f)

        with open(scaler_path, "r", encoding="utf-8") as f:
            scaler_data = json.load(f)

        try:
            import tensorflow as tf
            tflite_interpreter = tf.lite.Interpreter(model_path=str(model_path))
        except ImportError:
            import tflite_runtime.interpreter as tflite
            tflite_interpreter = tflite.Interpreter(model_path=str(model_path))

        tflite_interpreter.allocate_tensors()
        print("TFLite model loaded successfully")
        return True

    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return False


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"['']", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_term_occurrences(text: str, term: str) -> int:
    text = normalize_text(text)
    term = normalize_text(term)

    if not term:
        return 0

    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return len(re.findall(pattern, text))


def count_terms(text: str, terms: list) -> int:
    return sum(count_term_occurrences(text, term) for term in terms)


def analyze_lexicon(text: str) -> dict:
    urgency = count_terms(text, URGENCY_WORDS)
    security = count_terms(text, SECURITY_WORDS)
    pressure = count_terms(text, PRESSURE_WORDS)
    sensitive = count_terms(text, SENSITIVE_WORDS)
    authority = count_terms(text, AUTHORITY_WORDS)

    score = (
        2 * urgency +
        2 * security +
        2 * pressure +
        3 * sensitive +
        1 * authority
    )

    active_categories = sum([
        urgency > 0,
        security > 0,
        pressure > 0,
        sensitive > 0,
        authority > 0
    ])

    word_count = len(text.split()) if text else 0

    if word_count < 3:
        decision = "Indeterminate"
        is_vishing = False
    else:
        is_vishing = (score >= 4) or (active_categories >= 3)
        decision = "Fraudulent" if is_vishing else "Not Fraudulent"

    return {
        "decision": decision,
        "is_vishing": is_vishing,
        "score": score,
        "active_categories": active_categories,
        "word_count": word_count,
        "details": {
            "urgency_count": urgency,
            "security_count": security,
            "pressure_count": pressure,
            "sensitive_count": sensitive,
            "authority_count": authority
        }
    }


def generate_ngrams(text: str, ngram_range: List[int] = [1, 2]) -> List[str]:
    words = text.split()
    ngrams = []
    min_n, max_n = ngram_range

    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i+n]))

    return ngrams


def compute_tfidf_features(text: str) -> np.ndarray:
    global tfidf_data

    vocabulary = tfidf_data["vocabulary"]
    idf_values = tfidf_data["idf"]
    ngram_range = tfidf_data.get("ngram_range", [1, 2])
    num_features = tfidf_data["num_text_features"]

    normalized = normalize_text(text)
    ngrams = generate_ngrams(normalized, ngram_range)
    term_counts = Counter(ngrams)

    tfidf_vector = np.zeros(num_features, dtype=np.float32)
    total_ngrams = max(len(ngrams), 1)

    for term, count in term_counts.items():
        if term in vocabulary:
            idx = vocabulary[term]
            if idx < num_features and idx < len(idf_values):
                tf = count / total_ngrams
                tfidf_vector[idx] = tf * idf_values[idx]

    norm = np.linalg.norm(tfidf_vector)
    if norm > 0:
        tfidf_vector = tfidf_vector / norm

    return tfidf_vector


def compute_numeric_features() -> np.ndarray:
    global scaler_data

    num_features = scaler_data["num_numeric_features"]
    median_values = scaler_data.get("median_fill_values", {})
    feature_names = scaler_data.get("feature_names", [])

    features = np.zeros(num_features, dtype=np.float32)

    for i, name in enumerate(feature_names):
        if i < num_features:
            features[i] = median_values.get(name, 0.0)

    mean = np.array(scaler_data["mean"], dtype=np.float32)
    scale = np.array(scaler_data["scale"], dtype=np.float32)
    scale = np.where(scale == 0, 1, scale)

    return (features - mean) / scale


def predict_tflite(text: str) -> Optional[dict]:
    global tflite_interpreter, metadata

    if tflite_interpreter is None:
        return None

    try:
        tfidf_features = compute_tfidf_features(text)
        numeric_features = compute_numeric_features()

        all_features = np.concatenate([numeric_features, tfidf_features])
        expected_dim = metadata["input_dim"]

        if len(all_features) < expected_dim:
            all_features = np.pad(all_features, (0, expected_dim - len(all_features)))
        else:
            all_features = all_features[:expected_dim]

        input_data = all_features.reshape(1, -1).astype(np.float32)

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        tflite_interpreter.set_tensor(input_details[0]["index"], input_data)
        tflite_interpreter.invoke()

        output = tflite_interpreter.get_tensor(output_details[0]["index"])
        raw_score = float(output[0][0])

        if raw_score < 0 or raw_score > 1:
            probability = 1 / (1 + np.exp(-raw_score))
        else:
            probability = raw_score

        threshold = metadata.get("threshold", 0.52)
        is_vishing = probability >= threshold

        return {
            "is_vishing": is_vishing,
            "probability": probability,
            "threshold": threshold,
            "raw_score": raw_score
        }

    except Exception as e:
        print(f"TFLite prediction error: {e}")
        return None


def analyze_hybrid(text: str) -> dict:
    lexicon_result = analyze_lexicon(text)
    tflite_result = predict_tflite(text)

    word_count = len(text.split()) if text else 0

    if word_count < 3:
        return {
            "is_vishing": False,
            "decision": "Indeterminate",
            "confidence": 0.0,
            "vishing_probability": 0.0,
            "risk_level": "indeterminate",
            "method": "insufficient_text",
            "lexicon": lexicon_result,
            "tflite": None
        }

    lexicon_vishing = lexicon_result["is_vishing"]
    tflite_vishing = tflite_result["is_vishing"] if tflite_result else False
    tflite_prob = tflite_result["probability"] if tflite_result else 0.0

    is_vishing = lexicon_vishing or tflite_vishing

    lexicon_score = lexicon_result["score"]
    lexicon_prob = min(0.95, 0.3 + (lexicon_score / 15.0)) if lexicon_vishing else max(0.05, lexicon_score / 20.0)

    if tflite_result:
        if lexicon_score >= 6:
            combined_prob = 0.6 * lexicon_prob + 0.4 * tflite_prob
        else:
            combined_prob = 0.4 * lexicon_prob + 0.6 * tflite_prob
    else:
        combined_prob = lexicon_prob

    if lexicon_vishing and not tflite_vishing:
        combined_prob = max(combined_prob, 0.55)

    if combined_prob >= 0.80:
        risk_level = "high"
    elif combined_prob >= 0.52 or is_vishing:
        risk_level = "medium"
    else:
        risk_level = "low"

    decision = "Fraudulent" if is_vishing else "Not Fraudulent"

    return {
        "is_vishing": is_vishing,
        "decision": decision,
        "confidence": combined_prob if is_vishing else (1 - combined_prob),
        "vishing_probability": combined_prob,
        "risk_level": risk_level,
        "method": "hybrid",
        "lexicon": lexicon_result,
        "tflite": tflite_result
    }


def transcribe_audio_local(audio_path: str, language: Optional[str] = None) -> dict:
    global whisper_model

    if whisper_model is None:
        raise ValueError("Whisper model not loaded")

    segments, info = whisper_model.transcribe(
        audio_path,
        language=language if language and language != "auto" else None,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    text_parts = [segment.text.strip() for segment in segments]
    full_text = " ".join(text_parts)

    return {
        "text": full_text,
        "language": info.language,
        "duration": info.duration
    }


class TranscriptionRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = None


class AnalyzeTextRequest(BaseModel):
    text: str


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: Optional[float] = None


class AnalysisResponse(BaseModel):
    is_vishing: bool
    decision: str
    confidence: float
    vishing_probability: float
    risk_level: str
    transcribed_text: Optional[str] = None
    timestamp: str
    method: Optional[str] = None
    detected_language: Optional[str] = None
    lexicon_details: Optional[dict] = None
    tflite_details: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 60)
    print("Starting Vishing Detection API v5.0 - HYBRID")
    print("=" * 60 + "\n")

    whisper_ok = load_whisper_model()
    tflite_ok = load_tflite_model()

    print(f"Whisper loaded: {whisper_ok}")
    print(f"TFLite loaded: {tflite_ok}")

    print("\n" + "=" * 60)
    print("HYBRID MODE: TFLite model + Lexicon analysis")
    print("=" * 60 + "\n")


@app.get("/")
async def root():
    return {
        "message": "Vishing Detection API is running",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "5.0.0",
        "mode": "HYBRID (TFLite + Lexicon)",
        "whisper_loaded": whisper_model is not None,
        "tflite_loaded": tflite_interpreter is not None,
        "requires_api_key": False
    }


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")

    try:
        audio_data = base64.b64decode(request.audio_base64)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        try:
            result = transcribe_audio_local(tmp_path, request.language)
            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                duration=result.get("duration")
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalyzeTextRequest):
    try:
        text = request.text.strip() if request.text else ""
        result = analyze_hybrid(text)

        return AnalysisResponse(
            is_vishing=result["is_vishing"],
            decision=result["decision"],
            confidence=result["confidence"],
            vishing_probability=result["vishing_probability"],
            risk_level=result["risk_level"],
            transcribed_text=text,
            timestamp=datetime.utcnow().isoformat(),
            method=result["method"],
            lexicon_details=result.get("lexicon"),
            tflite_details=result.get("tflite")
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/detect")
async def detect_vishing(request: TranscriptionRequest):
    try:
        transcription = await transcribe_audio(request)
        analysis_request = AnalyzeTextRequest(text=transcription.text)
        result = await analyze_text(analysis_request)

        return AnalysisResponse(
            is_vishing=result.is_vishing,
            decision=result.decision,
            confidence=result.confidence,
            vishing_probability=result.vishing_probability,
            risk_level=result.risk_level,
            transcribed_text=result.transcribed_text,
            timestamp=result.timestamp,
            method=result.method,
            detected_language=transcription.language,
            lexicon_details=result.lexicon_details,
            tflite_details=result.tflite_details
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/api/detect-file")
async def detect_vishing_file(file: UploadFile = File(...), language: Optional[str] = None):
    try:
        content = await file.read()
        audio_base64 = base64.b64encode(content).decode("utf-8")

        request = TranscriptionRequest(audio_base64=audio_base64, language=language)
        return await detect_vishing(request)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File detection failed: {str(e)}")


@app.post("/api/test-analyze")
async def test_analyze(text: str = ""):
    if not text:
        return {"error": "Provide 'text' parameter"}

    result = analyze_hybrid(text)

    return {
        "input_text": text,
        "normalized_text": normalize_text(text),
        "word_count": len(text.split()),
        "final_decision": result["decision"],
        "is_vishing": result["is_vishing"],
        "combined_probability": result["vishing_probability"],
        "risk_level": result["risk_level"],
        "lexicon_analysis": result["lexicon"],
        "tflite_analysis": result["tflite"]
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
