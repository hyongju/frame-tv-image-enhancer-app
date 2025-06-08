from sqlalchemy.orm import Session
from . import models, schemas

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(
        email=user.email,
        name=user.name,
        picture=user.picture,
        tier=user.tier # Defaults to 'free' if not provided
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user