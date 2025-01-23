from typing import Optional
from pydantic import BaseModel, ValidationError

# Example: Default values and optional fields

class Person(BaseModel):
    first_name: str  # Required field
    last_name: str  # Required field
    country: str = "USA"  # Default value
    age: Optional[int] = None  # Optional field with default None
    is_active: Optional[bool] = True  # Optional field with default True

# Valid test case: Providing only required fields
print("### Valid Test Case 1: Required Fields Only ###")
try:
    person = Person(first_name="Jane", last_name="Doe")
    print(person)
    # Expected output:
    # first_name='Jane', last_name='Doe', country='USA', age=None, is_active=True
except ValidationError as e:
    print("Unexpected error:", e)


# Valid test case: Providing some optional fields
print("\n### Valid Test Case 3: Some Optional Fields ###")
try:
    person = Person(
        first_name="Alice",
        last_name="Brown",
        age=25  # Providing age but not is_active or country
    )
    print(person)
    # Expected output:
    # first_name='Alice', last_name='Brown', country='USA', age=25, is_active=True
except ValidationError as e:
    print("Unexpected error:", e)

# Invalid test case: Missing required fields
print("\n### Invalid Test Case 1: Missing Required Fields ###")
try:
    person = Person(last_name="Doe")  # Missing first_name
except ValidationError as e:
    print("Error:", e)

# Invalid test case: Incorrect data type for age
print("\n### Invalid Test Case 2: Incorrect Data Type ###")
try:
    person = Person(
        first_name="Bob",
        last_name="Green",
        age="Twenty"  # Invalid type for age (should be int or None)
    )
except ValidationError as e:
    print("Error:", e)

# Combining default and optional fields in a different model
print("\n### Combining Default and Optional Fields in Another Model ###")

class Task(BaseModel):
    title: str  # Required
    completed: Optional[bool] = False  # Optional, defaults to False
    priority: Optional[int] = None  # Optional, defaults to None

# Valid test case: Required field only
try:
    task = Task(title="Submit assignment")
    print(task)
    # Expected output:
    # title='Submit assignment', completed=False, priority=None
except ValidationError as e:
    print("Unexpected error:", e)

# Valid test case: Providing all fields
try:
    task = Task(title="Write report", completed=True, priority=1)
    print(task)
    # Expected output:
    # title='Write report', completed=True, priority=1
except ValidationError as e:
    print("Unexpected error:", e)

# Invalid test case: Missing required title
try:
    task = Task(completed=True)
except ValidationError as e:
    print("Error:", e)
