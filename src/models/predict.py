from fastapi import FastAPI, Header, HTTPException
import torch
import prepare_data


app = FastAPI()

PATH = "./model009.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = prepare_data.model()
model.load_state_dict(torch.load(PATH, map_location = device))




@app.post("/predict/{comment}")
async def make_prediction(comment: str):
	model.eval()
	data = prepare_data.prepare_comment(comment)

	for i, (input_ids) in enumerate(data):
		response = model(input_ids)
		resp = torch.IntTensor.item(response.argmax())
	if resp == 0:
		return {"response": "Negative"}
	else:
		return {"response": "Positive"}

		
