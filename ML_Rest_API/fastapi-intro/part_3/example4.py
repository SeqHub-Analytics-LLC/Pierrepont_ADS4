from pydantic import BaseModel, field_serializer
class Prediction(BaseModel):
    class_name: str
    probability: float # Raw probability between 0 and 1

    @field_serializer("probability")
    def format_probability(cls, value):
        return f"{value * 100:.2f}%" # Convert to percentage
    
# Prediction output
prediction = Prediction(class_name="Cat", probability=0.87)
# Serialize prediction
print(prediction.dict())
print(prediction.json())
