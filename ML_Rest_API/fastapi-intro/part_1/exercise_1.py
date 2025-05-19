from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/products/{product_name}")
def get_product(product_name: str):
    if not product_name.isalpha():
        raise HTTPException(status_code=400, detail="Invalid product name. Must be a string.")
    
    return {"product_id": product_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("exercise_1:app", host = "0.0.0.0", port = 8000, reload=True)