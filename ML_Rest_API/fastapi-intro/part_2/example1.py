from pydantic import BaseModel

class Person(BaseModel):
    first_name: str
    last_name: str
    age: int

# Create a valid person instance
valid_person = Person(
    first_name="Jonathan",
    last_name="Lockwood", 
    age=6
)
print(valid_person)

# Create an invalid person instance: Invalid age type
try:
    invalid_person = Person(
        first_name="Jonathan",
        last_name="Lockwood",
        age="John"  
    )
except Exception as e:
    print("Error:", e)
