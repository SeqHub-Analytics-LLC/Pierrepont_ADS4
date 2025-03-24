class UserPrediction(BaseModel):
  user_id: int # Sensitive information
  prediction: float
  
class Config:
  fields = {'user_id': {'exclude': True}} # Exclude user_id from serialization
  
# User prediction
user_prediction = UserPrediction(user_id=101, prediction=95.5)

# Serialize prediction
print(user_prediction.dict())
print(user_prediction.json())
