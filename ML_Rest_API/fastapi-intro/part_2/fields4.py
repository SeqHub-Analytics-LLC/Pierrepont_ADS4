from pydantic import BaseModel, Field
from typing import List
from enum import Enum


# Grade levels using Enum
class GradeLevel(str, Enum):
    FRESHMAN = "Freshman"
    SOPHOMORE = "Sophomore"
    JUNIOR = "Junior"
    SENIOR = "Senior"


# Student model
class Student(BaseModel):
    name: str = Field(..., min_length=1, description="The name of the student.")
    age: int = Field(..., ge=5, le=18, description="Age must be between 5 and 18.")
    grade_level: GradeLevel = Field(..., description="The grade level of the student.")
    enrolled_courses: List[str] = Field(..., min_items=1, description="At least one course must be enrolled.")

# Valid input
student = Student(
    name="Alice",
    age=15,
    grade_level="Sophomore",
    enrolled_courses=["Math", "Science"],
)
print(student)

# Invalid input (raises a ValidationError)
try:
    invalid_student = Student(
        name="",
        age=20,
        grade_level="Graduate",
        enrolled_courses=[],
    )
except Exception as e:
    print(e)