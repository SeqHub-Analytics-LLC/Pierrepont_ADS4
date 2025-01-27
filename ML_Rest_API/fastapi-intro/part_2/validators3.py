from typing import List
from pydantic import BaseModel, validator

class Hobby(BaseModel):
    hobbies: List[str]

    @validator("hobbies")
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