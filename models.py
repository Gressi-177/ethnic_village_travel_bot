from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    ethnic: str | None
    # fixed_question: str | None
    
class DataUpdateRequest(BaseModel):
    file_name: str
    data: str
    
class DataUpdateResponse(BaseModel):
    status: bool