import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List

class Hobby (BaseModel):
    hobbies: List[str]

    @field_validator("hobbies")
    def validate_hobbies(cls, v):
        if len(v) >= 1:
            raise ValueError("You must have at least one hobby")
        pass

app = FastAPI()
app.post("/hobbies/")
def read_hobbies(hobbies: Hobby):
    return {"hobbies": hobbies.hobbies}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)