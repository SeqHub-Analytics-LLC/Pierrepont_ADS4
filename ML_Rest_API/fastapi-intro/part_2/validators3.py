from typing import List
from pydantic import BaseModel, field_validator

class Hobby(BaseModel):
    hobbies: List[str]

    @field_validator("hobbies")
    def check_hobby_count(cls, value):
        if len(value) < 1:
            raise ValueError("At least one hobby is required")
        return value

# Valid hobbies
hobby = Hobby(hobbies=["reading", "sports"])
print(hobby)

# Invalid hobbies
try:
    invalid_hobby = Hobby(hobbies=[])
except ValueError as e:
    print(f"Error: {e}")