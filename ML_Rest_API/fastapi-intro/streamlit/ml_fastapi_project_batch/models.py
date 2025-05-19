# app/models.py

from pydantic import BaseModel, field_validator, root_validator
from enum import Enum

# Enum for Sex (Male or Female)
class SexEnum(str, Enum):
    Male = "Male"
    Female = "Female"

# Enum for Equipment types
class EquipmentEnum(str, Enum):
    Raw = "Raw"
    Wraps = "Wraps"
    Single_ply = "Single-ply"
    Multi_ply = "Multi-ply"

# Enum for Model Types
class ModelType(str, Enum): 
    rf = "Random Forests"
    dtc = "Decision Trees"
    gbt = "Gradient Boosting"

# Input Data model
class InputFeatures(BaseModel):
    Sex: SexEnum
    Equipment: EquipmentEnum
    Age: int
    BodyweightKg: float
    BestSquatKg: float
    BestDeadliftKg: float

    # Age validator to ensure it is between 18 and 100
    @field_validator('Age')
    def validate_age(cls, v):
        if not (18 <= v <= 100):
            raise ValueError('Age must be between 18 and 100')
        return v

    # Field validators for weight features (kg validation)
    @field_validator('BestSquatKg', 'BestDeadliftKg')
    def validate_kg_positive(cls, v):
        if v <= 0:
            raise ValueError(f'{v} must be greater than 0')
        return v

    # validator for BodyweightKg
    @field_validator('BodyweightKg')
    def validate_bodyweight(cls, v):
        if v <= 0:
            raise ValueError('BodyweightKg must be greater than 0')
        return v
    

# Prediction Request Data model
class PredictionRequest(BaseModel):
    Features: InputFeatures
    model: ModelType
