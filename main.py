import uvicorn
import json
from src.core.base_model import Message
from src.features.chat_completion import completion_with_retrieval
from fastapi import FastAPI

app = FastAPI()

@app.post('/illuminusAI')
async def main(message: Message):
    try:
        with open('./history.json', "r") as json_file:
            history_chat = json.load(json_file)
    except:
        history_chat = {}
    response = completion_with_retrieval(question=message.question, history=history_chat)
    return {'response': response}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)