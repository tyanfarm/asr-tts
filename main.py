import os

_MAIN_PROCESS_GUARD = os.environ.get('_MAIN_PROCESS_STARTED') != 'true'

if _MAIN_PROCESS_GUARD:
    os.environ['_MAIN_PROCESS_STARTED'] = 'true'
else:
    print(f"[MAIN GUARD] Blocking re-import in child process {os.getpid()}")

import torch
from kokoro import KPipeline
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import multiprocessing
import io
import logging
from pathlib import Path
import base64
import numpy as np
import sys

import librosa
import soundfile as sf
import lameenc

from uvicorn import Config, Server
import signal
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PHONEME_MODEL_NAME = "tyanfarm/aimate-asr"
TTS_MODEL_NAME = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 16000

TTS_PIPELINE = None
ASR_MODEL_COMPONENTS = None

# ============================================
# PyInstaller Helper Functions
# ============================================

def get_base_path():
    """Get base path works both in dev and PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        return Path(sys._MEIPASS)
    else:
        # Running in normal Python
        return Path(__file__).parent

def get_model_path(model_name):
    """Find model from Hugging Face cache or local directory"""
    base_path = get_base_path()
    local_model_path = base_path / "models" / model_name.replace("/", "-")
    
    if local_model_path.exists():
        return str(local_model_path), True

    return model_name, False

# ============================================
# Model Loading Functions
# ============================================

def load_asr_model(model_name=PHONEME_MODEL_NAME):
    
    """Load ASR model into memory"""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        # else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    
    model_path, is_local = get_model_path(model_name)
    
    logger.info(f"[ASR] Loading model: {model_name}")
    if is_local:
        logger.info(f"[ASR] From local path: {model_path}")
        processor = Wav2Vec2Processor.from_pretrained(
            model_path,
            local_files_only=True
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            model_path,
            local_files_only=True
        ).to(device).eval()
    else:
        logger.info(f"[ASR] From Hugging Face cache")
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device).eval()
    
    logger.info(f"[ASR] Using device: {device}")
    
    return processor, model, device

def load_tts_model(model_name=TTS_MODEL_NAME):
    """Load TTS model into memory"""
    logger.info(f"[TTS] Loading TTS model: {model_name}")
    pipeline = KPipeline(lang_code='a', repo_id=model_name)
    
    try:
        device = next(pipeline.model.parameters()).device
        logger.info(f"[TTS] Model device: {device}")
    except Exception as e:
        logger.warning(f"Cannot check TTS device: {str(e)}")
    
    return pipeline

# ============================================
# Global Model Storage (Lazy Loading)
# ============================================
def get_tts_model():
    global TTS_PIPELINE
    if TTS_PIPELINE is None:
        TTS_PIPELINE = load_tts_model() 
    return TTS_PIPELINE

def get_asr_model():
    global ASR_MODEL_COMPONENTS
    if ASR_MODEL_COMPONENTS is None:
        # Return (processor, model, device)
        ASR_MODEL_COMPONENTS = load_asr_model()
    return ASR_MODEL_COMPONENTS

# ============================================
# Lifespan Event Handler
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*50)
    logger.info("🚀 Starting server - Pre-loading models...")
    
    # Pre-load models ngay từ đầu
    try:
        logger.info("⏳ Loading ASR model...")
        get_asr_model()  # Force load ASR
        logger.info("✅ ASR model loaded")
        
        logger.info("⏳ Loading TTS model...")
        get_tts_model()  # Force load TTS
        logger.info("✅ TTS model loaded")
        
        await warmup_models()
        
        logger.info("🎉 All models ready!")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down - Cleaning up resources...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
# ============================================
# Warmup Function
# ============================================

async def warmup_models():
    """Warmup models with test inference"""
    
    try:        
        # 1. Warmup TTS
        logger.info("  ↳ Testing TTS model...")
        pipeline = TTS_PIPELINE
        
        # Generate test audio
        test_text = "Hello how are you today"
        wav_bytes, _ = text_to_speech(test_text, voice="af_heart", pipeline=pipeline)

        logger.info("  ✓ TTS warmup successful")
        
        # 1. Warmup ASR
        logger.info("  ↳ Testing ASR model...")
        processor, model, device = ASR_MODEL_COMPONENTS
        
        phonemes = extract_phonemes(wav_bytes, processor, model, device)
    
        print(f"  ↳ ASR warmup output phonemes: {phonemes}")
        
        logger.info("  ✓ ASR warmup successful")
    except Exception as e:
        logger.error(f" ❌ Warmup failed: {e}")

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="Audio Processing API",
    description="ASR (Phoneme Extraction) and TTS (Text to Speech) API",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================
# ASR Endpoints
# ============================================

def extract_phonemes(audio_data: bytes, processor, model, device):
    """Extract phonemes from audio bytes"""
    # Load audio from bytes
    audio, _ = librosa.load(io.BytesIO(audio_data), sr=SAMPLE_RATE)
    
    # Process audio
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes = processor.batch_decode(predicted_ids)[0]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return phonemes.strip()

@app.post("/asr/extract")
async def asr_extract_phonemes(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)")
):
    """
    Extract phonemes from audio file
    """
    try:
        processor, model, device = ASR_MODEL_COMPONENTS
        
        # Read audio file
        audio_data = await audio.read()
        
        # Extract phonemes using models from app.state
        phonemes = extract_phonemes(
            audio_data,
            processor,
            model,
            device
        )
        
        return JSONResponse(content={
            "success": True,
            "phonemes": phonemes,
        })
        
    except Exception as e:
        logger.error(f"ASR Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# TTS Endpoints
# ============================================

def text_to_speech(text: str, voice: str, pipeline, speed: float = None):
    """Convert text to speech"""
    # Auto-adjust speed based on text length
    if speed is None:
        words = text.split(" ")
        speed = 1.0 if len(words) <= 4 else 0.8
    
    generator = pipeline(text, voice=voice, speed=speed)
    audio_pieces = []

    for gs, ps, audio_chunk in generator:
        audio_pieces.append(audio_chunk)

    if len(audio_pieces) > 0:
        full_audio = np.concatenate(audio_pieces)

        # Create WAV
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, full_audio, samplerate=24000, format='WAV')
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.getvalue()

        # Create MP3
        audio_int16 = (full_audio * 32767).astype(np.int16)
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(128)
        encoder.set_in_sample_rate(24000)
        encoder.set_channels(1)
        encoder.set_quality(2)
        
        mp3_bytes = encoder.encode(audio_int16.tobytes())
        mp3_bytes += encoder.flush()

        return wav_bytes, mp3_bytes
    else:
        return None, None

@app.post("/tts/generate")
async def tts_generate(
    request: Request,
    payload: dict = Body(..., example={"text": "Hello world", "voice": "af_heart", "speed": 1.0})
):
    """
    Generate speech from text
    """
    try:
        pipeline = TTS_PIPELINE
        
        # Extract parameters from body
        text = payload.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="'text' field is required")
        
        voice = payload.get("voice", "af_heart")
        speed = payload.get("speed")
        
        # Generate audio using pipeline
        wav_bytes, mp3_bytes = text_to_speech(
            text, 
            voice, 
            pipeline,
            speed
        )
        
        if not wav_bytes or not mp3_bytes:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        wav_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        mp3_base64 = base64.b64encode(mp3_bytes).decode('utf-8')
        
        return JSONResponse(content={
            "success": True,
            "wav_base64": wav_base64,
            "mp3_base64": mp3_base64
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Health Check
# ============================================

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "asr_model_loaded": hasattr(request.app.state, 'asr_model') and request.app.state.asr_model is not None,
        "tts_model_loaded": hasattr(request.app.state, 'tts_pipeline') and request.app.state.tts_pipeline is not None,
        "device": str(request.app.state.asr_device) if hasattr(request.app.state, 'asr_device') else "unknown"
    }

# ============================================
# Server runner function
# ============================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    logger.info(f"="*60)
    logger.info(f"MAIN PROCESS - PID: {os.getpid()}")
    logger.info(f"="*60)
    
    # Cách 1: Dùng uvicorn.Server để kiểm soát tuyệt đối
    try:        
        config = Config(
            app=app,
            host="0.0.0.0",
            port=8080,
            log_level="info",
            loop="asyncio",  # Force asyncio loop
            workers=1,  # Chỉ 1 worker
            reload=False
        )
        
        server = Server(config)
        
        def handle_exit(sig, frame):
            logger.info(f"Received exit signal ({sig}). Stopping server...")
            server.should_exit = True
            
        signal.signal(signal.SIGINT, handle_exit)  # Ctrl+C
        signal.signal(signal.SIGTERM, handle_exit)
        
        logger.info("Starting uvicorn server...")
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")