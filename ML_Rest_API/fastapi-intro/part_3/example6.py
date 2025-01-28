from datetime import datetime
from pydantic import BaseModel

class InputFeatures(BaseModel):
    size: float
    location: str
    num_rooms: int

class OutputPrediction(BaseModel):
    price: float
    request_id: str
    timestamp: datetime

# Input from client
input_data = {"size": 120.5, "location": "Downtown", "num_rooms": 3}

# Deserialize input
features = InputFeatures(**input_data)
print("Validated Input:", features.dict())

# Generate prediction
prediction = OutputPrediction(
    price=250000.75,
    request_id="12345",
    timestamp=datetime.now()
)

# Serialize output
print("Serialized Output:", prediction.json())