from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Wlcome to my Product Page!"}

@app.get("/products/{product_name}")
def get_product(p_name: str):
    return {"product_name": p_name}