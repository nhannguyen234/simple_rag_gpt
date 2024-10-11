from pydantic import BaseModel

class Message(BaseModel):
    question: str