from pydantic import BaseModel,validator

class Person(BaseModel):
    first_name: str
    last_name: str

    @validator("first_name", "last_name")
    def capitalize_name(cls, value):
        return value.title()  # Capitalize the first letter of each word

# Creating a person with lowercase names
person = Person(first_name="john", last_name="doe")
print(person)