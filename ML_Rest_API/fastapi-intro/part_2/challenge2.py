import uvicorn
from pydantic import BaseModel, Field, field_validator, BeforeValidator
from fastapi import FastAPI
from typing import List
from typing_extensions import Annotated
from enum import Enum

def isNiceString(v) -> str:
    if not v.isalpha():
        raise ValueError("Value must be a string")
    return v 

def validate_num(value: int) -> int:
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            raise ValueError("Age must be a positive number")
    elif value <= 0:
        raise ValueError("Age must be a positive number")
    
    return value

class grades(str, Enum):
    freshman = "Freshman"
    sophmore = "Sophomore"
    junior = "Junior"
    senior = "Senior"

class Student(BaseModel):
    first_name: Annotated[str, BeforeValidator(isNiceString)]
    last_name: Annotated[str, BeforeValidator(isNiceString)]
    age: Annotated[int, Field(..., gt=1, description="The age of the student"), BeforeValidator(validate_num)]
    grade_level: grades = Field(..., description="The grade level of the student")
    
    @field_validator("grade_level")
    @classmethod
    def validate_grade_level(cls, value: str):
        if value not in [grade.value for grade in grades]:
            raise ValueError(f"Grade level must be one of {[grade.value for grade in grades]}")
        return value

class Location(BaseModel):
    city: Annotated[str, BeforeValidator(isNiceString)]
    state: Annotated[str, BeforeValidator(isNiceString)]
    zip_code: Annotated[int, BeforeValidator(validate_num)]

class School(BaseModel):
    name: Annotated[str, BeforeValidator(isNiceString)]
    location: Location = Field(..., description="The location of the school")
    students: List[Student] = Field(..., min_items=1, description="The students of the school")

app = FastAPI()
def create_school(school: School):
    return {"name": school.name, "location": school.location, "students": school.students}
app.add_api_route("/schools/", create_school, methods=["POST"])

if __name__ == "__main__":
    uvicorn.run("challenge2:app", host="127.0.0.1", port=8000, reload=True)