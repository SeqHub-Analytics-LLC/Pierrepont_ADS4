from datetime import date
from pydantic import BaseModel, ValidationError

# Define Address model
class Address(BaseModel):
    street_address: str
    postal_code: str
    city: str
    country: str

# Define Person model with nested Address model
class Person(BaseModel):
    first_name: str
    last_name: str
    birthdate: date
    address: Address

# Valid test case: Correct input for both Person and Address
print("### Valid Test Case ###")
try:
    person = Person(
        first_name="John",
        last_name="Doe",
        birthdate="1991-01-01",  # Valid date string
        address={
            "street_address": "123 Elm Street",
            "postal_code": "12345",
            "city": "Springfield",
            "country": "US"
        }
    )
    print("Valid Person:", person)
except ValidationError as e:
    print("Unexpected error:", e)

# Invalid test case: Missing required field in Address
print("\n### Invalid Test Case 1: Missing 'country' in Address ###")
try:
    person = Person(
        first_name="Jane",
        last_name="Doe",
        birthdate="1992-02-02",
        address={
            "street_address": "456 Oak Avenue",
            "postal_code": "54321",
            "city": "Metropolis"
            # Missing "country"
        }
    )
except ValidationError as e:
    print("Error:", e)

# Invalid test case: Incorrect type for address
print("\n### Invalid Test Case 2: Incorrect Address Type ###")
try:
    person = Person(
        first_name="Alice",
        last_name="Smith",
        birthdate="1993-03-03",
        address="Not a valid address object"  # Address must be a dictionary or Address object
    )
except ValidationError as e:
    print("Error:", e)

# Invalid test case: Incorrect date format
print("\n### Invalid Test Case 3: Invalid Birthdate Format ###")
try:
    person = Person(
        first_name="Bob",
        last_name="Brown",
        birthdate="InvalidDate",  # Invalid date format
        address={
            "street_address": "789 Pine Lane",
            "postal_code": "67890",
            "city": "Gotham",
            "country": "US"
        }
    )
except ValidationError as e:
    print("Error:", e)

# Invalid test case: Empty fields
print("\n### Invalid Test Case 4: Empty Fields ###")
try:
    person = Person(
        first_name="",
        last_name="",
        birthdate="1994-04-04",
        address={
            "street_address": "",
            "postal_code": "",
            "city": "",
            "country": ""
        }
    )
except ValidationError as e:
    print("Error:", e)

# Valid test case: Using an Address object directly
print("\n### Valid Test Case 2: Using Address Object ###")
try:
    address = Address(
        street_address="987 Maple Street",
        postal_code="11111",
        city="Star City",
        country="Canada"
    )
    person = Person(
        first_name="Charlie",
        last_name="Johnson",
        birthdate="1985-05-05",
        address=address  # Pass Address object directly
    )
    print("Valid Person with Address Object:", person)
except ValidationError as e:
    print("Unexpected error:", e)
