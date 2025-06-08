from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from datetime import datetime, timedelta
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from fastapi.security import OAuth2PasswordBearer

import cv2
import io
import logging
import os
import numpy as np
import asyncio
import torch
import gc
import enum
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, UnidentifiedImageError
from math import sqrt
from typing import Optional

# --- Local Imports (Simulated for single file structure) ---
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer

# --- Imports for Secure Admin Dashboard ---
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request


# --- Load Environment Variables ---
# from dotenv import load_dotenv
# load_dotenv()

# ==============================================================================
# 1. DATABASE SETUP
# ==============================================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models (models.py) ---
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

# ==============================================================================
# 2. PYDANTIC SCHEMAS
# ==============================================================================
class UserBase(BaseModel):
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    tier: UserTier = UserTier.free

class UserCreate(UserBase):
    pass

class UserSchema(UserBase):
    id: int
    class Config:
        orm_mode = True

class TokenData(BaseModel):
    email: Optional[str] = None
    tier: Optional[str] = None

class GoogleToken(BaseModel):
    token: str

class AppToken(BaseModel):
    access_token: str
    token_type: str
    user: UserSchema

# ==============================================================================
# 3. DATABASE CRUD OPERATIONS
# ==============================================================================
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    db_user = User(
        email=user.email,
        name=user.name,
        picture=user.picture,
        tier=user.tier
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==============================================================================
# 4. AUTHENTICATION & DEPENDENCIES
# ==============================================================================
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 120))
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# --- Load Admin Credentials from Environment ---
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_SESSION_SECRET_KEY = os.getenv("ADMIN_SESSION_SECRET_KEY")

# --- Validate that all necessary environment variables are set ---
if not all([SECRET_KEY, GOOGLE_CLIENT_ID, ADMIN_USERNAME, ADMIN_PASSWORD, ADMIN_SESSION_SECRET_KEY]):
    raise ValueError("One or more required environment variables are not set. Check your .env file.")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/google")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    return user

async def require_premium_user(current_user: User = Depends(get_current_user)):
    if current_user.tier != UserTier.premium:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This feature requires a premium subscription."
        )
    return current_user

# ==============================================================================
# 5. FASTAPI APPLICATION & ENDPOINTS
# ==============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Image Processing API")
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:3131", "https://image.hyongju.com", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Config & State ---
NETSCALE = 3
ENHANCE_TIMEOUT_SECONDS = 120
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
TARGET_FINAL_WIDTH = 3840
TARGET_FINAL_HEIGHT = 2160
ESRGAN_TILE_SIZE = 512
ESRGAN_TILE_PAD = 10
upsampler = None
num_workers = 1 if torch.cuda.is_available() else (os.cpu_count() or 2)
thread_pool_executor = ThreadPoolExecutor(max_workers=num_workers)

@app.on_event("startup")
async def startup_event():
    global upsampler
    try:
        logger.info("Initializing RealESRGANer...")
        model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
        if not os.path.exists(model_path):
            logger.error(f"RealESRGAN model not found at {model_path}. Premium endpoint will fail.")
            return

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        gpu_id = 0 if torch.cuda.is_available() else None
        upsampler = RealESRGANer(
            scale=4, model_path=model_path, model=model,
            tile=ESRGAN_TILE_SIZE, tile_pad=ESRGAN_TILE_PAD, pre_pad=0,
            half=True if gpu_id is not None else False, gpu_id=gpu_id
        )
        device_name = upsampler.device if hasattr(upsampler, 'device') else 'CPU'
        logger.info(f"RealESRGANer initialized on device: {device_name}")
    except Exception as e:
        logger.error(f"Failed to initialize RealESRGANer: {e}", exc_info=True)

def resize_and_crop_to_fill(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    img_w, img_h = image.size
    if img_w == 0 or img_h == 0: return image
    target_aspect = target_w / target_h
    img_aspect = img_w / img_h
    if img_aspect >= target_aspect:
        scale_factor = target_h / img_h
        interim_h = target_h
        interim_w = int(round(img_w * scale_factor))
    else:
        scale_factor = target_w / img_w
        interim_w = target_w
        interim_h = int(round(img_h * scale_factor))
    resized_image = image.resize((interim_w, interim_h), Image.Resampling.LANCZOS)
    left = (interim_w - target_w) / 2
    top = (interim_h - target_h) / 2
    cropped_image = resized_image.crop((left, top, left + target_w, top + target_h))
    return cropped_image

def synchronous_enhancement_function(image_cv_data, current_upsampler, output_scale_factor):
    if current_upsampler is None:
        raise RuntimeError("Upsampler not initialized for premium enhancement.")
    try:
        processed_cv, _ = current_upsampler.enhance(image_cv_data, outscale=output_scale_factor)
        return processed_cv
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()

@app.get("/")
async def read_root():
    return {"message": "Image Processing API is running"}

@app.post("/auth/google", response_model=AppToken)
async def authenticate_google_user(google_token: GoogleToken, db: Session = Depends(get_db)):
    try:
        idinfo = id_token.verify_oauth2_token(google_token.token, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in Google token")

        db_user = get_user_by_email(db, email=email)
        if not db_user:
            user_in = UserCreate(email=email, name=idinfo.get("name"), picture=idinfo.get("picture"))
            db_user = create_user(db, user=user_in)
        
        access_token = create_access_token(data={"sub": db_user.email, "tier": db_user.tier.value})
        return {"access_token": access_token, "token_type": "bearer", "user": db_user}
    except ValueError as e:
        logger.error(f"Google token verification error: {e}")
        raise HTTPException(status_code=403, detail=f"Invalid Google token: {e}")

@app.get("/users/me", response_model=UserSchema)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/process-image/", summary="Free Tier: Resize to 4K")
async def process_image_free(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    logger.info(f"Processing FREE request for user: {current_user.email}")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File size exceeds limit of {MAX_FILE_SIZE_MB}MB.")
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        final_image = resize_and_crop_to_fill(pil_image, TARGET_FINAL_WIDTH, TARGET_FINAL_HEIGHT)
        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error during free processing for {current_user.email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during image resizing.")

@app.post("/process-image-premium/", summary="Premium Tier: AI Upscale to 4K")
async def process_image_premium(file: UploadFile = File(...), premium_user: User = Depends(require_premium_user)):
    if upsampler is None:
        raise HTTPException(status_code=503, detail="AI upscaling service is currently unavailable.")
    
    logger.info(f"Processing PREMIUM request for user: {premium_user.email}")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File size exceeds limit of {MAX_FILE_SIZE_MB}MB.")
    
    try:
        pil_image = Image.open(io.BytesIO(contents))
        img_cv = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        loop = asyncio.get_event_loop()
        enhance_task = partial(synchronous_enhancement_function, img_cv, upsampler, NETSCALE)
        processed_cv = await asyncio.wait_for(
            loop.run_in_executor(thread_pool_executor, enhance_task),
            timeout=ENHANCE_TIMEOUT_SECONDS
        )
        gc.collect()
        if processed_cv is None:
            raise HTTPException(status_code=500, detail="Image enhancement failed.")
        processed_pil = Image.fromarray(cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB))
        final_image = resize_and_crop_to_fill(processed_pil, TARGET_FINAL_WIDTH, TARGET_FINAL_HEIGHT)
        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timed out.")
    except Exception as e:
        logger.error(f"Error during premium processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during image enhancement.")

@app.post("/process-image-premium-resize/", summary="Premium Tier: Fast Resize to 4K")
async def process_image_premium_resize(
    file: UploadFile = File(...),
    premium_user: User = Depends(require_premium_user)
):
    logger.info(f"Processing PREMIUM FAST RESIZE request for user: {premium_user.email}")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File size exceeds limit of {MAX_FILE_SIZE_MB}MB.")
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        final_image = resize_and_crop_to_fill(pil_image, TARGET_FINAL_WIDTH, TARGET_FINAL_HEIGHT)
        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot identify image file.")
    except Exception as e:
        logger.error(f"Error during premium fast resize for {premium_user.email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during image resizing.")

# ==============================================================================
# 6. ADMIN DASHBOARD SETUP (sqladmin)
# ==============================================================================

# --- Create a simple Authentication Class for the Admin Dashboard ---
class MyAdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        username, password = form["username"], form["password"]

        # Check the provided username and password against the environment variables
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            # Store a simple token in the session to indicate the user is logged in.
            request.session.update({"token": "admin_logged_in"})
            return True

        return False

    async def logout(self, request: Request) -> bool:
        # Clear the session token on logout.
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        # This function is called on every request to the admin panel.
        # It checks if the session token exists.
        return "token" in request.session

# Instantiate the authentication backend using the secret key from the environment.
authentication_backend = MyAdminAuth(secret_key=ADMIN_SESSION_SECRET_KEY)

# Create an Admin instance and attach it to the FastAPI app, the database engine,
# and the new authentication backend.
admin = Admin(app, engine, authentication_backend=authentication_backend)


# This class defines how the User model will be displayed in the admin dashboard.
class UserAdmin(ModelView, model=User):
    # Define which columns to show in the list view of users.
    column_list = [User.id, User.email, User.name, User.tier, User.created_at]

    # Allow inline editing for the 'tier' column from the list view.
    column_editable_list = [User.tier]

    # Add a search bar that searches by the 'email' and 'name' columns.
    column_searchable_list = [User.email, User.name]

    # Add a filter to easily see only 'free' or 'premium' users.
    column_filters = [User.tier]

    # A friendly name for the model in the sidebar.
    name_plural = "Application Users"
    
    # Set the number of items displayed per page.
    page_size = 50
    
    # You can also set a custom icon for the sidebar
    icon = "fa-solid fa-users"


# Add the configured User view to the admin dashboard.
admin.add_view(UserAdmin)