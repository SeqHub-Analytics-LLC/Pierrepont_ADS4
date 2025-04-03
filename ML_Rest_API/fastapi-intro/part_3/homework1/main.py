from enum import Enum
from typing import List, Optional
from datetime import date, datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, ValidationInfo

app = FastAPI()

# Enums
class MenuItemCategory(str, Enum):
    APPETIZER = "appetizer"
    MAIN_COURSE = "main course"
    DESSERT = "dessert"
    BEVERAGE = "beverage"
    SIDE = "side"
    SANDWICH = "sandwich"

class OrderType(str, Enum):
    DELIVERY = "delivery"
    PICKUP = "pickup"
    DINE_IN = "dine-in"

class OrderStatus(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    OUT_FOR_DELIVERY = "out_for_delivery"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

# Models
class MenuItem(BaseModel):
    name: str = Field(..., min_length=1)
    category: MenuItemCategory
    price: float = Field(..., gt=0)
    available: bool = True
    id: int = Field(..., gt=0)

class OrderItem(BaseModel):
    item_id: int
    quantity: int = Field(..., gt=0)

class Order(BaseModel):
    customer_name: str
    email: str
    order_type: OrderType
    delivery_address: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    items: List[OrderItem]
    scheduled_date: Optional[date] = None
    status_updated_at: Optional[date] = None

    @field_validator("email", mode="before")
    @classmethod
    def validate_email(cls, value: str) -> str:
        import re
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, value):
            raise ValueError("Invalid email format. Please provide a valid email address.")
        return value

    @field_validator("delivery_address", mode="before")
    @classmethod
    def validate_delivery_address(cls, value: Optional[str], info: ValidationInfo) -> Optional[str]:
        if value == OrderType.DELIVERY and not info.data.get("delivery_address"):
            raise ValueError("Delivery address is required for delivery orders.")
        return value

    @field_validator("scheduled_date", "status_updated_at", mode="after")
    @classmethod
    def validate_dates(cls, value: Optional[date]) -> Optional[date]:
        if value and value < date.today():
            raise ValueError("Dates must be today or in the future.")
        return value

    @field_validator("scheduled_date", "status_updated_at", mode="after")
    @classmethod
    def format_dates(cls, value: Optional[date]) -> Optional[str]:
        if value:
            return value.strftime("%d-%m-%Y")
        return value

# In-Memory Data
menu_items: List[MenuItem] = [
    MenuItem(name="Pizza", category=MenuItemCategory.MAIN_COURSE, price=12.50, id=101),
    MenuItem(name="Salad", category=MenuItemCategory.APPETIZER, price=8.00, id=102),
    MenuItem(name="Ice Cream", category=MenuItemCategory.DESSERT, price=5.00, id=103),
    MenuItem(name="Coke", category=MenuItemCategory.BEVERAGE, price=2.00, id=104),
]

orders: List[Order] = [
    Order(
        customer_name="John Doe",
        email="john.doe@example.com",
        order_type=OrderType.DELIVERY,
        delivery_address="123 Main St",
        items=[
            OrderItem(item_id=101, quantity=2),
            OrderItem(item_id=102, quantity=1),
        ],
        scheduled_date=date.today(),
    ),
    Order(
        customer_name="Jane Smith",
        email="jane.smith@example.com",
        order_type=OrderType.PICKUP,
        items=[
            OrderItem(item_id=103, quantity=3),
            OrderItem(item_id=104, quantity=1),
        ],
        scheduled_date=date.today(),
    ),
    Order(
        customer_name="Alice Johnson",
        email="alice.johnson@example.com",
        order_type=OrderType.DINE_IN,
        items=[
            OrderItem(item_id=101, quantity=2),
        ],
        scheduled_date=date.today(),
    ),
]

# Endpoints
@app.get("/menu_items")
async def get_menu_items(category: Optional[MenuItemCategory] = None, available: Optional[bool] = None):
    result = menu_items
    if category:
        result = [item for item in result if item.category == category]
    if available is not None:
        result = [item for item in result if item.available == available]
    return result

@app.post("/menu_items", response_model=MenuItem)
async def create_menu_item(item: MenuItem) -> MenuItem:
    menu_items.append(item)
    return item

@app.get("/menu_items/{item_id}")
async def get_menu_item(item_id: int) -> MenuItem:
    for item in menu_items:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Menu item not found")

@app.post("/place_order", response_model=Order)
async def place_order(order: Order):
    orders.append(order)
    return order

@app.get("/orders/{customer_name}", response_model=List[Order])
async def get_orders_by_customer(customer_name: str):
    customer_orders = [order for order in orders if order.customer_name == customer_name]
    if not customer_orders:
        raise HTTPException(status_code=404, detail="No orders found for this customer.")
    return customer_orders

@app.post("/update_order_status/{order_id}")
async def update_order_status(order_id: int, status: OrderStatus):
    if order_id < 0 or order_id >= len(orders):
        raise HTTPException(status_code=404, detail="The provided order doesn’t exist.")
    order = orders[order_id]
    valid_transitions = {
        OrderStatus.PENDING: [OrderStatus.PREPARING, OrderStatus.CANCELLED],
        OrderStatus.PREPARING: [OrderStatus.OUT_FOR_DELIVERY, OrderStatus.CANCELLED],
        OrderStatus.OUT_FOR_DELIVERY: [OrderStatus.COMPLETED],
    }
    if status not in valid_transitions.get(order.status, []):
        raise HTTPException(status_code=400, detail="Invalid status transition.")
    order.status = status
    return {"message": "Order successfully updated."}

@app.get("/orders_by_date", response_model=List[Order])
async def get_orders_by_date(start_date: Optional[date] = None, end_date: Optional[date] = None):
    filtered_orders = orders
    if start_date:
        filtered_orders = [order for order in filtered_orders if order.scheduled_date and order.scheduled_date >= start_date]
    if end_date:
        filtered_orders = [order for order in filtered_orders if order.scheduled_date and order.scheduled_date <= end_date]
    return filtered_orders

@app.post("/cancel_order/{order_id}")
async def cancel_order(order_id: int):
    if order_id < 0 or order_id >= len(orders):
        raise HTTPException(status_code=404, detail="The provided order doesn’t exist.")
    order = orders[order_id]
    if order.status == OrderStatus.COMPLETED:
        return {"message": "Cannot cancel a completed order."}
    order.status = OrderStatus.CANCELLED
    return {"message": "Order cancelled."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)