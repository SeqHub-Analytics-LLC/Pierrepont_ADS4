from enum import Enum
from fastapi import FastAPI, Path

app = FastAPI()

# Limit User Types
class UserRole(str, Enum):
    admin = "admin"
    standard = "standard"
    guest = "guest"

@app.get("/roles/{type}")
def get_role(type: UserRole):
    return {"role": type}

# Restrict Numeric IDs
@app.get("/products/{id}")
def get_product(id: int = Path(..., gt=100)):
    return {"product_id": id}

# Serial Number Validator
@app.get("/validate-serial/{serial}")
def validate_serial(serial: str = Path(..., regex=r"^[A-Z]{3}-\d{4}-[A-Z]{2}$")):
    return {"serial_number": serial, "status": "Valid"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("exercise_2:app", host="0.0.0.0", port=8000, reload=True)
