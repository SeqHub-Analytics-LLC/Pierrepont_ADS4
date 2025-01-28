from pydantic import BaseModel

class HouseFeatures(BaseModel):
    size: float  # In square meters
    location: str
    num_rooms: int

# Input data
features = HouseFeatures(size=120.5, location="Downtown", num_rooms=3)

# Serialize to dictionary
print(features.dict())

# Serialize to JSON
print(features.json())