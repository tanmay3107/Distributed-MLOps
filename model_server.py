# model_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
import asyncio

# --- 1. Data Validation Schemas ---
class EnrichedTransaction(BaseModel):
    tx_id: str
    user_id: str
    amount: float
    current_city: str
    # Historical features appended by our Redis Feature Store
    avg_spend_30d: float
    risk_score: float
    last_location: str

class PredictionResponse(BaseModel):
    tx_id: str
    is_fraud: bool
    confidence: float
    inference_time_ms: float

# --- 2. Mock Neural Network ---
class FraudDetectionModel:
    def __init__(self):
        self.model_loaded = False

    async def load_weights(self):
        """Simulates loading heavy .pt weights into GPU VRAM on startup."""
        print("🧠 Loading model weights into VRAM...")
        await asyncio.sleep(1.0)
        self.model_loaded = True
        print("✅ Model loaded and ready for inference.")

    def predict_batch(self, transactions: List[EnrichedTransaction]) -> List[dict]:
        """
        Simulates the forward pass of a PyTorch/XGBoost model.
        Instead of real matrix multiplication, we use a heuristic mock.
        """
        results = []
        for tx in transactions:
            start_time = time.time()
            
            # Simulated model logic looking for anomalies
            is_anomaly = False
            confidence = 0.95
            
            # Rule 1: Sudden massive spend compared to history
            if tx.amount > (tx.avg_spend_30d * 10) and tx.avg_spend_30d > 0:
                is_anomaly = True
                confidence = 0.99
                
            # Rule 2: Impossible travel (New York to Paris in 10 minutes)
            if tx.current_city != tx.last_location and tx.last_location != "UNKNOWN":
                is_anomaly = True
                confidence = 0.85
                
            inference_time_ms = (time.time() - start_time) * 1000
            
            results.append({
                "tx_id": tx.tx_id,
                "is_fraud": is_anomaly,
                "confidence": confidence,
                "inference_time_ms": inference_time_ms
            })
            
        return results

# --- 3. FastAPI Application ---
app = FastAPI(title="Fraud Detection Inference Server")
model = FraudDetectionModel()

@app.on_event("startup")
async def startup_event():
    """Triggered the moment the API boots up."""
    await model.load_weights()

@app.post("/predict", response_model=List[PredictionResponse])
async def predict_fraud(batch: List[EnrichedTransaction]):
    """
    The main inference endpoint. 
    Accepts a batch of fully enriched transactions and returns predictions.
    """
    if not model.model_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading into memory.")
        
    if len(batch) > 32:
        raise HTTPException(status_code=413, detail="Batch size exceeds maximum limit of 32 to prevent OOM.")
        
    print(f"📥 Received batch of {len(batch)} transactions for inference.")
    
    # Run the model forward pass
    predictions = model.predict_batch(batch)
    
    return predictions

# --- Quick Test Instructions ---
if __name__ == "__main__":
    print("🚀 To run this server, execute the following in your terminal:")
    print("   uvicorn model_server:app --host 0.0.0.0 --port 8000 --reload")