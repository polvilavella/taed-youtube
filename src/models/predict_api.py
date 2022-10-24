
""" Script that predicts the sentiment of a comment sent through the API. """

from fastapi import FastAPI
import torch
from data import prepare_data_api


app = FastAPI()

PATH = "./model009.pt"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = prepare_data_api.model()
model.load_state_dict(torch.load(PATH, map_location = DEVICE))


@app.post("/predict/{comment}")
async def make_prediction(comment: str):
    """ Predict the sentiment of the comment with the uploaded model. """
    model.eval()
    data = prepare_data_api.prepare_comment(comment)

    for _, (input_ids) in enumerate(data):
        response = model(input_ids)
        resp = int(response.argmax().item())
    if resp == 0:
        return {"response": "Negative"}
    return {"response": "Positive"}
