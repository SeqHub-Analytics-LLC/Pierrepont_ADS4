import uvicorn
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from fastapi import FastAPI

class roles (str, Enum):
    fighter = "Fighter"
    wizard = "Wizard"
    cleric = "Cleric"
    rogue = "Rogue"

class Character(BaseModel):
    name: str
    age: int = Field(..., gt=10, lt=100, description="The age of the character")
    class_type: roles = Field(..., description="The class of the character")
    abilities: List[str] = Field(..., min_items=1, description="The abilities of the character")

app = FastAPI()
@app.post("/characters/")
def create_character(character: Character):
    return {"name": character.name, "age": character.age, "class": character.class_type, "abilities": character.abilities}

if __name__ == "__main__":
    uvicorn.run("challenge1:app", host="127.0.0.1", port=8000, reload=True)

