from pydantic import BaseModel
class Prediction(BaseModel):
    price: float # Predicted price in dollars
    
# Model prediction
prediction = Prediction(price=250000.75)
# Serialize prediction
print(prediction.dict())
print(prediction.json())
