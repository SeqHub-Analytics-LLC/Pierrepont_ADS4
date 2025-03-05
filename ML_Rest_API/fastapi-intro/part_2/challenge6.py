import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

class loanpurpose(str, Enum):
    home = "home"
    car = "car"
    education = "education"

class LoanApplication(BaseModel):
    name: str
    income: float
    loan_amount: float
    purpose: loanpurpose

    @field_validator("name")
    def validate_name(cls, v):
        if len(v) < 0:
            raise ValueError("Name must be at least 1 character long")
        pass

    @field_validator("income")
    def validate_income(cls, v):
        if v <1000:
            raise ValueError("Income must be greater than 1000")
        pass

    @model_validator()
    def validate_loan_amount(Self):
        if Self.loan_amount >= 5*Self.income:
            raise ValueError("Loan amount must be less than 5 times the income")
        pass

app = FastAPI()
app.post("/loans/")
def apply_for_loan(loan: LoanApplication):
    return {"name": loan.name, "income": loan.income, "loan_amount": loan.loan_amount, "purpose": loan.purpose}

if __name__ == "__main__":
    uvicorn.run(app, host="12.0.0.1", port=8000)