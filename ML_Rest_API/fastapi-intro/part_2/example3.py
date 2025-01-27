### **Working with Dates**

from datetime import date
from pydantic import BaseModel, ValidationError

class Person(BaseModel):
    first_name: str
    last_name: str
    birthdate: date

try:
    person = Person(
        first_name="John",
        last_name="Doe",
        birthdate="1991-13-42"  # Invalid date
    )
except ValidationError as e:
    print(e)

#### Example of Valid Date:
person = Person(
    first_name="John",
    last_name="Doe",
    birthdate="1991-01-01"
)
print(person)
