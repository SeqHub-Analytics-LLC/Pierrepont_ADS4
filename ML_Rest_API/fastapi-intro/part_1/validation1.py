from enum import Enum
from fastapi import FastAPI
# Define an Enum for user types
class UserType(str, Enum):    
    STANDARD = "standard"   
    ADMIN = "admin"
# Create the FastAPI app
app = FastAPI()
# Use the Enum as a path parameter type
@app.get("/users/{type}/{id}")
def get_user(type: UserType, id: int):    
    return {"type": type, "id": id}