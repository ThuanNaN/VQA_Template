from pydantic import BaseModel

class OutputTemplate(BaseModel):
    template: str

