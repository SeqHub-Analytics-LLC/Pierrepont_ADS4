from typing import List
from pydantic import BaseModel, Field


class HobbyList(BaseModel):
    hobbies: List[str] = Field(..., min_items=1, description="You must list at least one hobby.")

# Valid example
valid_hobbies = HobbyList(hobbies=["reading", "gaming"])

# Invalid example (raises a ValidationError)
invalid_hobbies = HobbyList(hobbies=[])