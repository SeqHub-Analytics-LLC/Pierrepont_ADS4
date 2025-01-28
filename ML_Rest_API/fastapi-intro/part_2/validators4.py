from pydantic import BaseModel, EmailStr, field_validator

class User(BaseModel):
    email: EmailStr

    @field_validator("email")
    def check_email_domain(cls, value):
        if not value.endswith("@example.com"):
            raise ValueError("Email must end with @example.com")
        return value.lower()  # Transform email to lowercase

# Valid email
user = User(email="John.Doe@EXAMPLE.com")
print(user)

# Invalid email
try:
    invalid_user = User(email="john.doe@gmail.com")
except ValueError as e:
    print(f"Error: {e}")