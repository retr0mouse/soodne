from pydantic import BaseModel

class JobStatus(BaseModel):
    message: str

    class Config:
        from_attributes = True