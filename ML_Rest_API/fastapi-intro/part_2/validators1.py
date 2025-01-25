from pydantic import BaseModel, validator

class Person(BaseModel):
    first_name: str
    last_name: str
    age: int

    @validator("age")
    def validate_age(cls, value):
        if value <= 0:
            raise ValueError("Age must be a positive number")
        return value

# Valid age
person = Person(first_name="Alice", last_name="Smith", age=25)
print(person)

# Invalid age
try:
    invalid_person = Person(first_name="Alice", last_name="Smith", age=-5)
except ValueError as e:
    print(f"Error: {e}")

