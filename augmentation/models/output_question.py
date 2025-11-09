from pydantic import BaseModel

class OutputQuestion(BaseModel):
    question: str
    answer: str

class ListQuestions(BaseModel):
    questions: list[OutputQuestion]