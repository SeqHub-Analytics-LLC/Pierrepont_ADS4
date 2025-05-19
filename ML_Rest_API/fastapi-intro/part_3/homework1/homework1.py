from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field, validator, field_serializer
from typing import List, Optional
from enum import Enum
from datetime import date, datetime
import re

app = FastAPI()

# Enums
class MenuCategory(str, Enum):
    appetizer = "appetizer"
    main_course = "main_course"
    dessert = "dessert"
    beverage = "beverage"
    sides = "sides"
    sandwich = "sandwich"

class OrderType(str, Enum):
    delivery = "delivery"
    pickup = "pickup"
    dine_in = "dine-in"

class OrderStatus(str, Enum):
    pending = "Pending"
    preparing = "Preparing"
    out_for_delivery = "Out for Delivery"
    completed = "Completed"
    cancelled = "Cancelled"

# Pydantic Models
class MenuItem(BaseModel):
    id: int
    name: str
    category: MenuCategory
    price: float = Field(..., gt=0)
    available: bool = True

class OrderItem(BaseModel):
    item_id: int
    quantity: int = Field(..., ge=1)

class Order(BaseModel):
    id: int
    customer_name: str
    email: str
    order_type: OrderType
    delivery_address: Optional[str] = None
    status: OrderStatus = OrderStatus.pending
    items: List[OrderItem]
    scheduled_date: Optional[date] = Field(default=None, description="Date of the game (YYYY-MM-DD)")
    status_updated_at: Optional[date] = Field(default=None, description="Date of the game (YYYY-MM-DD)")

    @validator("email")
    def validate_email(cls, v):
        pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email format")
        return v

    @validator("delivery_address")
    def check_delivery_address(cls, v, values):
        if values.get("order_type") == OrderType.delivery and not v:
            raise ValueError("Delivery address is required for delivery orders")
        return v

    @validator("scheduled_date", "status_updated_at")
    def validate_dates(cls, v):
        if v and v < date.today():
            raise ValueError("Date must be today or in the future")
        return v

    @field_serializer("scheduled_date", "status_updated_at")
    def serialize_date(self, value):
        if value:
            return value.strftime("%d-%m-%Y")
        return None

# Response Models
class MenuItemsResponse(BaseModel):
    menu_items: List[MenuItem]

class OrdersResponse(BaseModel):
    orders: List[Order]

class MessageResponse(BaseModel):
    message: str

# In-memory "databases"
menu_items = [
    MenuItem(id=1, name="Fries", category="sides", price=3.5, available = True),
    MenuItem(id=2, name="Burger", category="main_course", price=8.0, available=True),
    MenuItem(id=3, name="Cola", category="beverage", price=2.0, available=True),
    MenuItem(id=4, name="Salad", category="appetizer", price=4.5, available=False),
]

order_items = [
    OrderItem(item_id=1, quantity=2),
    OrderItem(item_id=2, quantity=1),
    OrderItem(item_id=3, quantity=3),
    OrderItem(item_id=4, quantity=1),
    OrderItem(item_id=2, quantity=2)
]

orders = [
    Order(id=1, customer_name="Alice", email="alice@example.com", order_type="pickup", status = "Completed", items=[order_items[0], order_items[1]], scheduled_date=date(2025, 4, 1), status_updated_at=date(2025, 4, 1)),
    Order(id=2, customer_name="Bob", email="bob@example.com", order_type="delivery", status = "Preparing", delivery_address="123 Watkins Street, new Haven, CT", items=[order_items[2]], scheduled_date=date(2025, 4, 23)),
    Order(id=3, customer_name="Alex", email="alex@example.com", order_type="dine-in", status = "Pending", items=[order_items[3], order_items[4]], scheduled_date=date(2025, 4, 3))
]

#home root
@app.get("/")
def home():
    return {"Message": "Welcome to Tee Tog's Cafe!"}

# Menu Endpoints
@app.get("/menu_items")
def get_menu_items(category: Optional[MenuCategory] = None, available: Optional[bool] = None):
    result = menu_items
    if category:
        result = [item for item in result if item.category == category]
    if available is not None:
        result = [item for item in result if item.available == available]
    response = MenuItemsResponse(menu_items=result)
    return response.json()

@app.post("/menu_items")
def add_menu_item(item: MenuItem):
    menu_items.append(item)
    response = MessageResponse(message="Menu item successfully added")
    return response.json()

@app.get("/menu_items/{item_id}")
def get_menu_item_by_id(item_id: int):
    for item in menu_items:
        if item.id == item_id:
            return {"Response": item}
    raise HTTPException(status_code=404, detail="Menu item not found")


# Order Endpoints
@app.post("/place_order")
def place_order(order: Order):
    orders.append(order)
    response = MessageResponse(message="Order successfully placed")
    return response.json()

@app.get("/orders/{customer_name}")
def get_orders_by_customer(customer_name: str):
    customer_orders = [o for o in orders if o.customer_name.lower() == customer_name.lower()]
    if not customer_orders:
        raise HTTPException(status_code=404, detail="No orders found for this customer")
    response = OrdersResponse(orders=customer_orders)
    return response.json()

@app.post("/update_order_status/{order_id}")
def update_order_status(order_id: int, new_status: OrderStatus):
    for order in orders:
        if order.id == order_id:
            valid_transitions = {
                OrderStatus.pending: [OrderStatus.preparing, OrderStatus.cancelled],
                OrderStatus.preparing: [OrderStatus.out_for_delivery, OrderStatus.cancelled],
                OrderStatus.out_for_delivery: [OrderStatus.completed],
            }
            if new_status not in valid_transitions.get(order.status, []):
                raise HTTPException(status_code=400, detail="Invalid status transition")
            order.status = new_status
            order.status_updated_at = date.today()
            response = MessageResponse(message="Order successfully updated")
            return response.json()
    raise HTTPException(status_code=404, detail="The provided order doesn’t exist")

@app.get("/orders_by_date")
def get_orders_by_date(start_date:  date = Query(None, description="Filter order starting from this date (YYYY-MM-DD)"),
                        end_date:  date = Query(None, description="Filter order ending at this date (YYYY-MM-DD)")):
    result = []
    for order in orders:
        if order.scheduled_date:
            if start_date and order.scheduled_date < start_date:
                continue
            if end_date and order.scheduled_date > end_date:
                continue
        result.append(order)
    response = OrdersResponse(orders=result)
    return response.json()

@app.post("/cancel_order/{order_id}")
def cancel_order(order_id: int):
    for order in orders:
        if order.id == order_id:
            if order.status == OrderStatus.completed:
                response = MessageResponse(message="Cannot cancel a completed order")
                return response.json()
            order.status = OrderStatus.cancelled
            response = MessageResponse(message="Order cancelled")
            return response.json()
    raise HTTPException(status_code=404, detail="The provided order doesn’t exist")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("homework1:app", host="0.0.0.0", port=8000, reload=True)