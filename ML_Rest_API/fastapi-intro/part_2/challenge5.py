import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    name: str
    price: float
    stock: int

    @field_validator("name")
    def validate_name(cls, v):
        if len(v) < 0:
            raise ValueError("Name must be at least 1 character long")
        pass

    @field_validator("price")
    def validate_price(cls, v):
        if v < 0:
            raise ValueError("Price must be greater than 0")
        pass

    @field_validator("stock")
    def validate_stock(cls, v):
        if v < 0:
            raise ValueError("Stock must be greater than 0")
        pass

app = FastAPI()
app.post("/products/")
def create_product(product: Product):
    return {"name": product.name, "price": product.price, "stock": product.stock}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)