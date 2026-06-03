from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ...,
        description="Model feature names mapped to their values.",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("features must not be empty")

        for feature_name in value:
            if not isinstance(feature_name, str) or not feature_name.strip():
                raise ValueError("feature names must be non-empty strings")

        return value


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the customer is high risk.",
    )
    predicted_class: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary risk label derived from the probability.",
    )
    model_uri: str = Field(
        ...,
        description="MLflow registry URI used to load the model.",
    )
