"""### **Challenge 4: Inventory Management System (Using Classes and Magic Methods)**

#### **Objective**:
Create a simple inventory management system for a store. Each item in the inventory should:
1. Have attributes like `name`, `price`, and `quantity`.
2. Support operations like:
   - Adding two items of the same type (combine their quantities).
   - Comparing two items to check if they are equal (same name and price).
   - Displaying the item details using a string representation (using `__str__`).

#### **Requirements**:
1. Implement a class called `Item` with attributes `name`, `price`, and `quantity`.
2. Add the following magic methods:
   - `__add__`: Allows adding two items of the same type to combine their quantities.
   - `__eq__`: Compares two items for equality (based on `name` and `price`).
   - `__str__`: Returns a string representation of the item in the format: `"Item(name=Apple, price=1.5, quantity=10)"`.
3. Test the class by creating a few items and performing the operations.

---

#### **Example Input/Output**:

##### **Example 1**:
```python
item1 = Item("Apple", 1.5, 10)
item2 = Item("Apple", 1.5, 5)
item3 = Item("Banana", 0.5, 20)

# Adding items
combined_item = item1 + item2
print(combined_item)  # Output: Item(name=Apple, price=1.5, quantity=15)

# Comparing items
print(item1 == item2)  # Output: True
print(item1 == item3)  # Output: False

# String representation
print(item1)  # Output: Item(name=Apple, price=1.5, quantity=10)
```


"""

class Item:
    """Class representing an inventory item with name, price, and quantity."""
    
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

    def __add__(self, other):
        """Combines the quantity of two items if they have the same name and price."""
        if isinstance(other, Item) and self.name == other.name and self.price == other.price:
            return Item(self.name, self.price, self.quantity + other.quantity)
        else:
            raise ValueError("Items must have the same name and price to be combined.")

    def __eq__(self, other):
        """Checks if two items are equal based on name and price."""
        if isinstance(other, Item):
            return self.name == other.name and self.price == other.price
        return False

    def __str__(self):
        """Returns a string representation of the item."""
        return f"Item(name={self.name}, price={self.price}, quantity={self.quantity})"


# Test cases
item1 = Item("Apple", 1.5, 10)
item2 = Item("Apple", 1.5, 5)
item3 = Item("Banana", 0.5, 20)

combined_item = item1 + item2
print(combined_item)
print(item1 == item2)
print(item1 == item3)