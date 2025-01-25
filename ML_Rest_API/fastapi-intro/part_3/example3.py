from datetime import datetime

class Metadata(BaseModel):
    request_id: str
    timestamp: datetime

class Response(BaseModel):
    prediction: float
    metadata: Metadata

# Nested data
response = Response(
    prediction=250000.75,
    metadata=Metadata(request_id="12345", timestamp=datetime.now())
)

# Serialize response
print(response.dict())
print(response.json())