from sqlalchemy import Column, Integer, String, DateTime, func, Enum as SQLEnum
from .database import Base
import enum

class UserTier(enum.Enum):
    free = "free"
    premium = "premium"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)
    picture = Column(String)
    tier = Column(SQLEnum(UserTier), nullable=False, default=UserTier.free)
    created_at = Column(DateTime(timezone=True), server_default=func.now())