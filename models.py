from pydantic import BaseModel, EmailStr, Field
from pydantic_extra_types.phone_numbers import PhoneNumber


class ContactInfo(BaseModel):
    name: str = Field(
        description="A full name consisting of first and last parts, example: John Doe"
    )
    phoneNumber: PhoneNumber
    email: EmailStr
    policyNumber: str | None
