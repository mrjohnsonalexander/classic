from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
import tensorflow
from tensorflow import keras
import keras_nlp
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
import os
from pydantic import BaseModel


api = KaggleApi()
api.authenticate()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
keras.config.set_floatx("float16")
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_1.1_instruct_2b_en", load_weights=False)

app = FastAPI()

logging.basicConfig(filename='main.log', level=logging.INFO)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class HealthCheck(BaseModel):
      """Response model to validate and return when performing a health check."""

      status: str = "OK"

@app.get(
     "/health",
     tags=["healthcheck"],
     summary="Perform a Health Check",
     response_description="Return HTTP Status Code 200 (OK)",
     status_code=status.HTTP_200_OK,
     response_model=HealthCheck,
)

def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
       Returns:
          HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")

@app.get("/generate")
async def generate():
    
    try:
        generated = gemma_lm.generate("What is the meaning of life?", max_length=30)
    except:
        logger.info("Logged generation error")

    return generated