##fastapi

from fastapi import FastAPI, Request
from pydantic import BaseModel  ##to validate the requests
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from fastapi.templating import Jinja2Templates  ###to render the UI templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


#initialize the app
app = FastAPI(title="Text Summarization APP", description="This is a text summarization app using T5 model", version="1.0")



model= T5ForConditionalGeneration.from_pretrained("./final_model")
tokenizer= T5Tokenizer.from_pretrained("./final_model")




if torch.backends.mps.is_available():
    device= torch.device("mps")
elif torch.cuda.is_available():
    device= torch.device("cuda")
else:
    device= torch.device("cpu")

model.to(device)





###templating
templates= Jinja2Templates(directory=".")


##i/p schema/format (for string dialogue)
class DialogueInput(BaseModel):
    dialogue: str





def clean_data(text):
    text = re.sub(r'\r\n+', ' ', text)
    text= re.sub(r'\s+', ' ', text) 
    text= re.sub(r'<.*?>', ' ', text) 
    text= text.strip()
    return text
    




def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        targets = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=200,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary





##API endpts. ---> / => root or home , /summarize => summary request
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary= summarize_dialogue(dialogue_input.dialogue)
    return  {"summary": summary}