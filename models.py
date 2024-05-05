from pydantic import BaseModel, EmailStr


class UserPolicy(BaseModel):
    name: str
    phoneNumber: str
    email: EmailStr
    policyNumber: str | None
