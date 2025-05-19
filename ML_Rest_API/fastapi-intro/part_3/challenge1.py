from pydantic import BaseModel, field_serializer

class Product(BaseModel):
    name: str
    price: float
    category: str
    discount: float = 0.0

    @field_serializer("price")
    def format_price(cls, value):
        return f"{value:.2f}"

product_data = {
    "name": "Laptop",
    "price": 1500.0,
    "category": "Electronics"
}

product = Product(**product_data)
print(product.json())