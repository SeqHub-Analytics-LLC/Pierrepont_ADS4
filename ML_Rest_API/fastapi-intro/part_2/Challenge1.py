from fastapi import FastAPI, Path, Query, HTTPException
from enum import Enum
from pydantic import BaseModel, Field

app = FastAPI()

class Type(str, Enum):
    wizard = "Wizard"
    warrior = "Warrior"
    archer = "Archer"
    healer = "Healer"

class Character(BaseModel):
    name: str 
    age: int = Field(..., ge=10, le=100, description="Age has to be greater than or equal to 10 but less than or equal to 100")
    class_type: Type
    abilities: str = Field(..., min_items = 1, description="Must have at least one ability")

#character = Character(name = "Simon", age = 15, class_type = "Wizard", abilities = ["Flight"])
 
#print(character)


@app.post("/charcaters/")
def get_character(character: Character = Path(...,)):
    return {"character": character}

#if __name__ == "__main__":
    #import uvicorn
    #uvicorn.run("validation3:app", host="0.0.0.0", port=8000, reload=True)


