import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import List

class Address(BaseModel):
    street: str
    city: str
    
    @field_validator("city")
    def validate_city_length(cls, v):
        if len(v) < 3:
            raise ValueError("City name must be at least 3 characters long")
        pass

class UserProfile(BaseModel):
    name: str
    age: int
    address: Address

app = FastAPI()
app.post("/users/")
def create_user_profile(user: UserProfile):
    return {"name": user.name, "age": user.age, "address": user.address}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)