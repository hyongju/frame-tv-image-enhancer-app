from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import cv2
import glob
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError, ImageFile
import io
import logging
import os
from datetime import datetime, timedelta
from typing import Union, Optional
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.utils import RealESRGANer
from functools import partial
from math import sqrt, ceil, floor # <--- IMPORT FLOOR
# For Google OIDC
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# For our own JWTs
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

NETSCALE = 3
ENHANCE_TIMEOUT_SECONDS = 60
MAX_IMAGE_PIXELS = 1_000_000
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

TARGET_CROP_WIDTH = 3840
TARGET_CROP_HEIGHT = 2160

# --- Aspect Ratio Constants ---
ASPECT_RATIO_16_9 = 16 / 9
ASPECT_RATIO_9_16 = 9 / 16


if not GOOGLE_CLIENT_ID:
    raise ValueError("GOOGLE_CLIENT_ID not set in environment variables")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY not set in environment variables")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Processing API")

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost",
    "https://image.hyongju.com",
    "capacitor://localhost",
    "http://localhost",
    "ionic://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class TokenData(BaseModel):
    email: Optional[str] = None

class GoogleToken(BaseModel):
    token: str

class User(BaseModel):
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None

class AppToken(BaseModel):
    access_token: str
    token_type: str
    user: User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/google")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
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
    
    user_info_from_jwt = payload.get("user_info", {}) 
    user = User(email=email, name=user_info_from_jwt.get("name"), picture=user_info_from_jwt.get("picture"))
    return user

upsampler = None
thread_pool_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 2)

@app.on_event("startup")
async def startup_event():
    global upsampler
    try:
        logger.info("Initializing RealESRGANer...")
        model_native_scale = 4 
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_native_scale)
        model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
        
        if not os.path.exists(model_path):
            logger.error(f"RealESRGAN model not found at {model_path}.")
            return

        upsampler = RealESRGANer(
            scale=model_native_scale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,tile_pad=10,pre_pad=0,half=True,gpu_id=None 
        )
        logger.info("RealESRGANer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RealESRGANer during startup: {e}", exc_info=True)
        upsampler = None

@app.get("/")
async def read_root(): return {"message": "Image Processing API is running!"}

@app.post("/auth/google", response_model=AppToken)
async def authenticate_google_user(google_token: GoogleToken):
    # (Authentication logic remains the same)
    try:
        CLOCK_SKEW_SECONDS = 60
        idinfo = id_token.verify_oauth2_token(
            google_token.token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=CLOCK_SKEW_SECONDS
        )
        logger.info(f"Google ID Token verified for email: {idinfo.get('email')}")
        user_email = idinfo.get("email")
        if not user_email:
            raise HTTPException(status_code=400, detail="Email not found in Google token")
        
        user_data = User(email=user_email, name=idinfo.get("name"), picture=idinfo.get("picture"))
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        app_access_token = create_access_token(
            data={"sub": user_email, "user_info": user_data.dict()}, 
            expires_delta=access_token_expires
        )
        return {"access_token": app_access_token, "token_type": "bearer", "user": user_data}
    except ValueError as e:
        logger.error(f"Google token verification error: {e}")
        raise HTTPException(status_code=403, detail=f"Invalid Google token: {e}")
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Authentication failed")


def synchronous_enhancement_function(image_cv_data, current_upsampler, output_scale_factor):
    if current_upsampler is None:
        raise RuntimeError("Upsampler not initialized for synchronous enhancement.")
    processed_cv, _ = current_upsampler.enhance(image_cv_data, outscale=output_scale_factor)
    return processed_cv

@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    global upsampler
    if upsampler is None:
        logger.error(f"Upscaling attempt by {current_user.email} but upsampler not available.")
        raise HTTPException(status_code=503, detail="Image upscaling service is currently unavailable.")

    logger.info(f"Processing image request for user: {current_user.email}, file: {file.filename}")

    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded by {current_user.email}: {file.content_type}")
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image type.")

    try:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE_BYTES:
            logger.warning(f"File size limit exceeded: {len(contents)} bytes for {file.filename}")
            raise HTTPException(status_code=413, detail=f"Image file size exceeds limit of {MAX_FILE_SIZE_MB}MB.")

        pil_image_processing = None
        original_pil_format_name = "PNG"

        try:
            pil_image_processing = Image.open(io.BytesIO(contents))
            original_pil_format_name = pil_image_processing.format or "PNG"
            original_width, original_height = pil_image_processing.size
            current_pixels = original_width * original_height
            logger.info(f"Original: {original_width}x{original_height} ({current_pixels}px) for {file.filename}")

            # --- Step 1: Downsize if EXCEEDING MAX_IMAGE_PIXELS ---
            if current_pixels > MAX_IMAGE_PIXELS:
                logger.warning(f"Original exceeds MAX_IMAGE_PIXELS ({MAX_IMAGE_PIXELS}). Downsizing.")
                # (Downsizing logic as before)
                target_pixels_for_resize = MAX_IMAGE_PIXELS -1 if MAX_IMAGE_PIXELS > 0 else 1
                scale_ratio = sqrt(target_pixels_for_resize / current_pixels)
                new_width = max(1, int(original_width * scale_ratio))
                new_height = max(1, int(original_height * scale_ratio))
                pil_image_processing = pil_image_processing.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Downsized to: {pil_image_processing.width}x{pil_image_processing.height} ({pil_image_processing.width * pil_image_processing.height}px)")
            
            current_w, current_h = pil_image_processing.size

            # --- Step 2: PRE-UPSCALING IF NEEDED for RealESRGAN input ---
            if NETSCALE <= 0: logger.error("NETSCALE must be positive.")
            else:
                min_input_w_for_esrgan = ceil(TARGET_CROP_WIDTH / NETSCALE)
                min_input_h_for_esrgan = ceil(TARGET_CROP_HEIGHT / NETSCALE)
                projected_esrgan_w = current_w * NETSCALE
                projected_esrgan_h = current_h * NETSCALE
                needs_pre_upscale = (projected_esrgan_w < TARGET_CROP_WIDTH) or (projected_esrgan_h < TARGET_CROP_HEIGHT)

                if needs_pre_upscale:
                    logger.info(f"Image ({current_w}x{current_h}) projected to ({projected_esrgan_w}x{projected_esrgan_h}) by NETSCALE={NETSCALE} "
                                f"is insufficient for target crop ({TARGET_CROP_WIDTH}x{TARGET_CROP_HEIGHT}). Pre-upscaling for RealESRGAN.")
                    upscale_factor = 1.0
                    if current_w > 0 and current_w < min_input_w_for_esrgan: upscale_factor = max(upscale_factor, min_input_w_for_esrgan / current_w)
                    if current_h > 0 and current_h < min_input_h_for_esrgan: upscale_factor = max(upscale_factor, min_input_h_for_esrgan / current_h)
                    
                    if upscale_factor > 1.0:
                        new_pre_w = ceil(current_w * upscale_factor)
                        new_pre_h = ceil(current_h * upscale_factor)
                        pil_image_processing = pil_image_processing.resize((new_pre_w, new_pre_h), Image.Resampling.LANCZOS)
                        logger.info(f"Pre-upscaled to: {pil_image_processing.width}x{pil_image_processing.height}")
                    # (else: pre-upscale not needed or factor <= 1)
                # (else: no pre-upscale needed)
            
            # --- Step 2.5: CROP TO 16:9 & CONSTRAIN PIXELS IF EXCEEDING MAX_IMAGE_PIXELS (NEW) ---
            current_w_after_pre_upscale, current_h_after_pre_upscale = pil_image_processing.size
            current_pixels_after_pre_upscale = current_w_after_pre_upscale * current_h_after_pre_upscale

            if current_pixels_after_pre_upscale > MAX_IMAGE_PIXELS:
                logger.info(
                    f"Image after pre-upscale ({current_w_after_pre_upscale}x{current_h_after_pre_upscale}, {current_pixels_after_pre_upscale}px) "
                    f"exceeds MAX_IMAGE_PIXELS ({MAX_IMAGE_PIXELS}). Applying 16:9 constraint."
                )
                
                # Target 16:9 dimensions that fit strictly within MAX_IMAGE_PIXELS
                safe_max_pixels = MAX_IMAGE_PIXELS - 1 if MAX_IMAGE_PIXELS > 0 else 0
                h_target_fit, w_target_fit = 0, 0

                if safe_max_pixels >= 1: # Need at least 1 pixel to form any image
                    h_target_fit = floor(sqrt(safe_max_pixels * ASPECT_RATIO_9_16))
                    if h_target_fit > 0:
                        w_target_fit = floor(h_target_fit * ASPECT_RATIO_16_9)
                        while w_target_fit * h_target_fit > safe_max_pixels and h_target_fit > 0:
                            h_target_fit -= 1
                            if h_target_fit > 0: w_target_fit = floor(h_target_fit * ASPECT_RATIO_16_9)
                            else: w_target_fit = 0; break
                    h_target_fit = max(1, h_target_fit) # Ensure at least 1px height
                    w_target_fit = max(1, w_target_fit if h_target_fit > 0 else 1) # Ensure at least 1px width
                
                if w_target_fit == 0 or h_target_fit == 0: # Fallback if calculation failed (e.g. MAX_IMAGE_PIXELS too small)
                    logger.warning(f"MAX_IMAGE_PIXELS ({MAX_IMAGE_PIXELS}) too small for meaningful 16:9. Skipping 16:9 constraint for this step.")
                else:
                    logger.info(f"Target 16:9 dimensions to fit MAX_IMAGE_PIXELS: {w_target_fit}x{h_target_fit}")
                    
                    # Crop current image to 16:9 aspect ratio (as large as possible from current image)
                    img_aspect = current_w_after_pre_upscale / current_h_after_pre_upscale if current_h_after_pre_upscale > 0 else float('inf')
                    crop_w_intermediate, crop_h_intermediate = current_w_after_pre_upscale, current_h_after_pre_upscale

                    if img_aspect > ASPECT_RATIO_16_9: # Wider than 16:9, crop sides
                        crop_w_intermediate = floor(current_h_after_pre_upscale * ASPECT_RATIO_16_9)
                    elif img_aspect < ASPECT_RATIO_16_9: # Taller than 16:9, crop top/bottom
                        crop_h_intermediate = floor(current_w_after_pre_upscale * ASPECT_RATIO_9_16)
                    # else: already 16:9, no change to intermediate crop dimensions

                    crop_w_intermediate = max(1, min(current_w_after_pre_upscale, crop_w_intermediate))
                    crop_h_intermediate = max(1, min(current_h_after_pre_upscale, crop_h_intermediate))

                    if crop_w_intermediate > 0 and crop_h_intermediate > 0:
                        left = int((current_w_after_pre_upscale - crop_w_intermediate) / 2)
                        top = int((current_h_after_pre_upscale - crop_h_intermediate) / 2)
                        right = left + crop_w_intermediate
                        bottom = top + crop_h_intermediate
                        
                        # Ensure box is within bounds
                        left, top = max(0, left), max(0, top)
                        right, bottom = min(current_w_after_pre_upscale, right), min(current_h_after_pre_upscale, bottom)

                        if left < right and top < bottom :
                            img_cropped_to_16_9 = pil_image_processing.crop((left, top, right, bottom))
                            logger.info(f"Intermediate image cropped to 16:9 aspect: {img_cropped_to_16_9.width}x{img_cropped_to_16_9.height}")

                            # If this 16:9 image still too large, resize to target_fit dimensions
                            if img_cropped_to_16_9.width * img_cropped_to_16_9.height > MAX_IMAGE_PIXELS:
                                logger.info(f"Resizing 16:9 image to {w_target_fit}x{h_target_fit} to fit MAX_IMAGE_PIXELS.")
                                pil_image_processing = img_cropped_to_16_9.resize((w_target_fit, h_target_fit), Image.Resampling.LANCZOS)
                            else:
                                pil_image_processing = img_cropped_to_16_9 # Already fits
                            
                            logger.info(f"Image after 16:9 constraint: {pil_image_processing.width}x{pil_image_processing.height} ({pil_image_processing.width * pil_image_processing.height}px)")
                        else: logger.warning("Invalid crop box for 16:9 intermediate. Skipping.")
                    else: logger.warning("Cannot determine valid 16:9 intermediate crop dims. Skipping.")
            # --- End Step 2.5 ---

            # Convert (potentially resized/cropped multiple times) PIL Image to OpenCV format
            if pil_image_processing.mode == 'RGBA': img_cv = cv2.cvtColor(np.array(pil_image_processing), cv2.COLOR_RGBA2BGRA)
            elif pil_image_processing.mode == 'RGB': img_cv = cv2.cvtColor(np.array(pil_image_processing), cv2.COLOR_RGB2BGR)
            elif pil_image_processing.mode == 'L': img_cv = cv2.cvtColor(np.array(pil_image_processing), cv2.COLOR_GRAY2BGR)
            elif pil_image_processing.mode == 'P':
                pil_image_processing_converted = pil_image_processing.convert('RGBA')
                img_cv = cv2.cvtColor(np.array(pil_image_processing_converted), cv2.COLOR_RGBA2BGRA)
            else:
                pil_converted = pil_image_processing.convert('RGB')
                img_cv = cv2.cvtColor(np.array(pil_converted), cv2.COLOR_RGB2BGR)
        
        except UnidentifiedImageError: # (Error handling as before)
            logger.error(f"Pillow UnidentifiedImageError for {file.filename}", exc_info=True)
            raise HTTPException(status_code=400, detail="Cannot identify image file.")
        except Exception as e_prep:
            logger.error(f"Error preparing image {file.filename}: {e_prep}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Error preparing image: {str(e_prep)}")
        
        logger.info(f"Input for RealESRGAN: {img_cv.shape if hasattr(img_cv, 'shape') else 'N/A'}")

        # --- Step 3: RealESRGAN ENHANCEMENT ---
        loop = asyncio.get_event_loop()
        enhance_task_with_args = partial(synchronous_enhancement_function, img_cv, upsampler, NETSCALE)
        processed_image_cv = await asyncio.wait_for(
            loop.run_in_executor(thread_pool_executor, enhance_task_with_args),
            timeout=ENHANCE_TIMEOUT_SECONDS
        )
        logger.info(f"Enhancement output shape: {processed_image_cv.shape}")

        # Convert enhanced OpenCV image back to PIL
        if len(processed_image_cv.shape) == 3 and processed_image_cv.shape[2] == 4:
            processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image_cv, cv2.COLOR_BGRA2RGBA))
        elif len(processed_image_cv.shape) == 3 and processed_image_cv.shape[2] == 3:
            processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2RGB))
        else:
             logger.error(f"Unexpected RealESRGAN output shape: {processed_image_cv.shape}")
             raise HTTPException(status_code=500, detail="Image processing failed: unexpected enhancer output.")

        # --- Step 4: FINAL CROPPING to TARGET_CROP_WIDTH x TARGET_CROP_HEIGHT ---
        logger.info(f"Image before final crop: {processed_pil_image.width}x{processed_pil_image.height}")
        image_to_save = processed_pil_image
        if processed_pil_image.width >= TARGET_CROP_WIDTH and processed_pil_image.height >= TARGET_CROP_HEIGHT:
            # (Cropping logic as before)
            left = (processed_pil_image.width - TARGET_CROP_WIDTH) / 2
            top = (processed_pil_image.height - TARGET_CROP_HEIGHT) / 2
            right = left + TARGET_CROP_WIDTH
            bottom = top + TARGET_CROP_HEIGHT
            crop_box = (int(left), int(top), int(right), int(bottom))
            try:
                processed_pil_image_cropped = processed_pil_image.crop(crop_box)
                if processed_pil_image_cropped.width == TARGET_CROP_WIDTH and processed_pil_image_cropped.height == TARGET_CROP_HEIGHT:
                    image_to_save = processed_pil_image_cropped
                    logger.info(f"Successfully cropped to {image_to_save.width}x{image_to_save.height}.")
                else: # Should not happen if logic is correct and source image was large enough
                    image_to_save = processed_pil_image_cropped # Use it anyway
                    logger.warning(f"Cropped to {image_to_save.width}x{image_to_save.height}, not exact target. Using.")
            except Exception as e_crop:
                logger.error(f"Error during final crop: {e_crop}", exc_info=True)
                # image_to_save remains processed_pil_image (uncropped)
        else:
            logger.warning(f"Upscaled ({processed_pil_image.width}x{processed_pil_image.height}) too small for final crop ({TARGET_CROP_WIDTH}x{TARGET_CROP_HEIGHT}). Skipping.")
        
        logger.info(f"Final output dimensions: {image_to_save.width}x{image_to_save.height}")

        # --- Step 5: PREPARE RESPONSE ---
        img_byte_arr = io.BytesIO()
        output_format_str = original_pil_format_name.upper()
        if output_format_str == "JPEG" and image_to_save.mode in ['RGBA', 'LA', 'P']:
            image_to_save = image_to_save.convert('RGB')
        
        if output_format_str == "GIF" and hasattr(image_to_save, 'is_animated') and image_to_save.is_animated:
             image_to_save.save(img_byte_arr, format=output_format_str, save_all=True, duration=image_to_save.info.get('duration', 100), loop=image_to_save.info.get('loop', 0))
        else:
            image_to_save.save(img_byte_arr, format=output_format_str)
        
        img_byte_arr.seek(0)
        media_type_suffix = output_format_str.lower()
        if media_type_suffix == "jpeg": media_type_suffix = "jpg"
        media_type = f"image/{media_type_suffix}"

        logger.info(f"Successfully processed {file.filename}. Returning as {media_type}.")
        return StreamingResponse(img_byte_arr, media_type=media_type)

    except asyncio.TimeoutError: # (Error handling as before)
        logger.error(f"Timeout after {ENHANCE_TIMEOUT_SECONDS}s for {file.filename}", exc_info=True)
        raise HTTPException(status_code=504, detail=f"Processing timed out.")
    except HTTPException: raise
    except cv2.error as e_cv:
        logger.error(f"OpenCV error for {file.filename}: {e_cv}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OpenCV processing error: {str(e_cv)}")
    except Exception as e_main:
        logger.error(f"Generic error for {file.filename}: {e_main}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e_main)}")

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)): return current_user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)