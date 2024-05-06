from langchain_core.pydantic_v1 import BaseModel, Field


class ContactInfo(BaseModel):
    name: str = Field(
        description="A full name consisting of first and last parts, example: John Doe"
    )
    phoneNumber: str = Field(
        description="A phone number with area code, example: (858)-453-4100"
    )
    email: str = Field(description="A valid email format, example: tony@gmail.com")
    policyNumber: str | None
