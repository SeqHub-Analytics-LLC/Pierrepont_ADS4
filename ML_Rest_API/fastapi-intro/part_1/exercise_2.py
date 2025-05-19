from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/roles/{type}")
def validate_role(type: str):
    allowed_roles = {"admin", "standard", "guest"}
    if type not in allowed_roles:
        raise HTTPException(status_code=400, detail="Invalid role. Allowed roles: admin, standard, guest")
    return {"message": f"Role '{type}' is valid."}

@app.get("/products/{id}")
def validate_product_id(id: int):
    if id <= 100:
        raise HTTPException(status_code=400, detail="ID must be greater than 100")
    return {"message": f"Product ID '{id}' is valid."}

