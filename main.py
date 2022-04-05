from http import HTTPStatus
from typing import List, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI
from gramformer import Gramformer
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(1212)

# loading the model
gf = Gramformer(models=1, use_gpu=False)


class RequestModel(BaseModel):
    text: Union[str, List[str]]


# FastAPI app init
app = FastAPI(
    title="Grammer Corrector",
    description="An helping hand to correct any grammetical error",
    version="1.0",
)
# REST API for health check
@app.get("/")
def _index():
    """Health check"""
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }


@app.post("/api/grammar-correct/")
def correct(input: RequestModel):
    """REST API to correct grammatical errors."""
    correct_result = []
    if isinstance(input.text, str):
        return gf.correct(input.text, max_candidates=1)
    elif isinstance(input.text, list):
        correct_result = [
            gf.correct(sentence, max_candidates=1) for sentence in input.text
        ]
        return correct_result

