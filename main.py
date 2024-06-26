from fastapi import FastAPI, HTTPException, status, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import logging
import os
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


default_question = "How to develop a reliable large language model system?"

tokenizer = AutoTokenizer.from_pretrained(
    "/root/.cache/kagglehub/models/google/gemma/transformers/1.1-2b-it/1"
)

model = AutoModelForCausalLM.from_pretrained(
    "/root/.cache/kagglehub/models/google/gemma/transformers/1.1-2b-it/1",
    device_map="auto",
    torch_dtype=torch.float16,
)

app = FastAPI()

templates = Jinja2Templates(directory="/usr/share/applications/templates/")

logger = logging.basicConfig(filename='main.log', level=logging.INFO)

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
def generate(question=default_question):
    try:
        input_ids = tokenizer(question, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=512)
        generated = tokenizer.decode(outputs[0])
    except Exception as e:
        logger.info(e)
    return generated


@app.get("/form", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse(
        'index.html', context={
            'request': request,
            'generated': "Ask a question"})


@app.post("/form")
def form(request: Request, question: str = Form(...)):
    return templates.TemplateResponse(
        'index.html', context={
            'request': request,
            "generated": generate(question)})
