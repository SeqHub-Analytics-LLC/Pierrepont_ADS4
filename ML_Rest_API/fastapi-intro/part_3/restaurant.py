from fastapi import FastAPI, Path, Query, HTTPException
from pydantic import BaseModel, Field, validator, field_serializer
from enum import Enum
from datetime import date

app = FastAPI()

# =========================
# Menu Management
# =========================

class Category(str, Enum):
    appetizer = "Appetizer"
    main_course = "Main Course"
    dessert = "Dessert"
    beverage = "Beverage"
    sides = "Sides"
    sandwich = "Sandwich"

class MenuItem(BaseModel):
    name: str
    category: Category
    price: float = Field(..., gt=0, description="Price must be greater than 0")
    available: bool = Field(default=True)

class MenuItemResponse(BaseModel):
    menu_items: list[MenuItems]

menu_items = [
    MenuItem(name="Burger", category=Category.sandwich, price=8.99),
    MenuItem(name="Chicken Wings", category=Category.appetizer, price=5.49),
    MenuItem(name="Steak", category=Category.main_course, price=12.99),
    MenuItem(name="Chocolate Cake", category=Category.dessert, price=4.99)
]

@app.get("/menu_items")
def get_menu_items(
    category: Category = Query(None, description="Filter by category"),
    available: bool = Query(None, description="Filter by availability")
):
    filtered = menu_items
    if category:
        filtered = [item for item in filtered if item.category == category]
    if available is not None:
        filtered = [item for item in filtered if item.available == available]
    return filtered

@app.post("/menu_items")
def add_menu_item(menu_item: MenuItem):
    menu_items.append(menu_item)
    return {"message": "Menu item added successfully"}

@app.get("/menu_items/{item_id}")
def get_menu_item_by_id(item_id: int = Path(..., ge=0, description="Item ID must be a positive integer")):
    if item_id < len(menu_items):
        return menu_items[item_id]
    raise HTTPException(status_code=404, detail="Menu item not found")


# =========================
# Order Management
# =========================

class OrderType(str, Enum):
    delivery = "Delivery"
    pickup = "Pickup"
    dine_in = "Dine-in"

class OrderStatus(str, Enum):
    pending = "Pending"
    preparing = "Preparing"
    out_for_delivery = "Out for Delivery"
    completed = "Completed"
    cancelled = "Cancelled"

class OrderItem(BaseModel):
    item_id: int = Field(..., ge=0, description="Item id must be a least 0")
    quantity: int = Field(..., ge=1, description="Quantity must be at least 1")

class Order(BaseModel):
    customer_name: str
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", description="Valid email address")
    order_type: OrderType
    delivery_address: str = None
    status: OrderStatus = Field(default=OrderStatus.pending)
    items: list[OrderItem]
    scheduled_date: date = None
    status_updated_at: date = None

    @validator("delivery_address", always=True)
    def check_delivery_address(cls, v, values):
        if values.get('order_type') == OrderType.delivery and not v:
            raise ValueError("Delivery address is required for delivery orders")
        return v

    @validator("scheduled_date", "status_updated_at")
    def validate_dates(cls, v):
        if v and v < date.today():
            raise ValueError("Dates must not be in the past")
        return v

    @field_serializer("scheduled_date")
    def serialize_scheduled_date(self, value: date) -> str:
        return value.strftime("%d-%m-%Y") if value else None

    @field_serializer("status_updated_at")
    def serialize_status_updated_at(self, value: date) -> str:
        return value.strftime("%d-%m-%Y") if value else None

orders = [
    Order(customer_name="Jane Doe", email="jane.doe@example.com", order_type=OrderType.dine_in, items=[OrderItem(item_id=0, quantity=2)], status=OrderStatus.pending),
    Order(customer_name="John Smith", email="john.smith@example.com", order_type=OrderType.delivery, delivery_address="123 Main St", items=[OrderItem(item_id=1, quantity=1)], status=OrderStatus.preparing),
    Order(customer_name="Alice Johnson", email="alice.johnson@example.com", order_type=OrderType.pickup, items=[OrderItem(item_id=2, quantity=3)], status=OrderStatus.out_for_delivery)
]

@app.post("/place_order")
def place_order(order: Order):
    orders.append(order)
    return {"message": "Order placed successfully"}

@app.get("/orders/{customer_name}")
def get_orders_by_customer(customer_name: str):
    customer_orders = [order for order in orders if order.customer_name.lower() == customer_name.lower()]
    if not customer_orders:
        raise HTTPException(status_code=404, detail="No orders found for this customer")
    return customer_orders.json()

@app.post("/update_order_status/{order_id}")
def update_order_status(order_id: int, new_status: OrderStatus):
    if order_id < len(orders):
        order = orders[order_id]
        if order.status == OrderStatus.completed:
            raise HTTPException(status_code=400, detail="Cannot update a completed order")
        
        if order.status == OrderStatus.pending and new_status not in [OrderStatus.preparing, OrderStatus.cancelled]:
            raise HTTPException(status_code=400, detail="Invalid status transition")
        if order.status == OrderStatus.preparing and new_status not in [OrderStatus.out_for_delivery, OrderStatus.cancelled]:
            raise HTTPException(status_code=400, detail="Invalid status transition")
        if order.status == OrderStatus.out_for_delivery and new_status != OrderStatus.completed:
            raise HTTPException(status_code=400, detail="Invalid status transition")
        
        order.status = new_status
        return {"message": "Order successfully updated"}
    
    raise HTTPException(status_code=404, detail="The provided order doesn't exist")

@app.get("/orders_by_date")
def get_orders_by_date(start_date: date = Query(None), end_date: date = Query(None)):
    filtered_orders = orders
    if start_date:
        filtered_orders = [order for order in filtered_orders if order.scheduled_date >= start_date]
    if end_date:
        filtered_orders = [order for order in filtered_orders if order.scheduled_date <= end_date]
    return filtered_orders

@app.post("/cancel_order/{order_id}")
def cancel_order(order_id: int):
    if order_id < len(orders):
        order = orders[order_id]
        if order.status == OrderStatus.completed:
            raise HTTPException(status_code=400, detail="Cannot cancel a completed order")
        order.status = OrderStatus.cancelled
        return {"message": "Order cancelled"}
    
    raise HTTPException(status_code=404, detail="Order not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("restaurant:app", host="0.0.0.0", port=8000, reload=True)
