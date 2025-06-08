from pydantic import BaseModel
from typing import Optional
from .models import UserTier

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    tier: Optional[str] = None

class GoogleToken(BaseModel):
    token: str

# --- User Schemas ---
class UserBase(BaseModel):
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    tier: UserTier = UserTier.free

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    class Config:
        orm_mode = True

class AppToken(BaseModel):
    access_token: str
    token_type: str
    user: User