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
from math import sqrt, ceil, floor

# For Google OIDC
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# For our own JWTs
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np

import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- GPU/Memory Management Imports ---
import torch  # For PyTorch operations, including CUDA management
import gc     # For Python's garbage collector

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

NETSCALE = 3 # This is the 'outscale' for RealESRGANer, not the model's native scale
ENHANCE_TIMEOUT_SECONDS = 60
MAX_IMAGE_PIXELS = 2_000_000 # Max pixels for image BEFORE RealESRGAN
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# --- Target Resolution for Final Output (Resize & Crop to Fill) ---
TARGET_FINAL_WIDTH = 3840
TARGET_FINAL_HEIGHT = 2160


# --- Aspect Ratio Constants ---
ASPECT_RATIO_16_9 = 16 / 9
ASPECT_RATIO_9_16 = 9 / 16

# --- RealESRGAN Tiling Configuration ---
ESRGAN_TILE_SIZE = 200 
ESRGAN_TILE_PAD = 10     

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
num_workers = 1 if torch.cuda.is_available() else (os.cpu_count() or 2)
thread_pool_executor = ThreadPoolExecutor(max_workers=num_workers)
logger.info(f"ThreadPoolExecutor initialized with max_workers={num_workers}")


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
            upsampler = None 
            return

        current_gpu_id = None
        if torch.cuda.is_available():
            logger.info(f"CUDA available. GPU Count: {torch.cuda.device_count()}. Using GPU:0")
            current_gpu_id = 0 
        else:
            logger.info("CUDA not available. Using CPU for RealESRGANer.")

        upsampler = RealESRGANer(
            scale=model_native_scale, 
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=ESRGAN_TILE_SIZE,        
            tile_pad=ESRGAN_TILE_PAD,     
            pre_pad=0,
            half=True if current_gpu_id is not None else False, 
            gpu_id=current_gpu_id
        )
        
        if hasattr(upsampler, 'device'):
             logger.info(f"RealESRGANer initialized. Model device: {upsampler.device}")
        else:
             logger.info(f"RealESRGANer initialized (device attribute not found, assuming based on gpu_id: {current_gpu_id}).")
        logger.info(f"RealESRGANer configuration: tile_size={ESRGAN_TILE_SIZE}, tile_pad={ESRGAN_TILE_PAD}, half_precision={'True on GPU' if current_gpu_id is not None else 'False on CPU'}")

    except Exception as e:
        logger.error(f"Failed to initialize RealESRGANer during startup: {e}", exc_info=True)
        upsampler = None 

@app.get("/")
async def read_root(): return {"message": "Image Processing API is running!"}

@app.post("/auth/google", response_model=AppToken)
async def authenticate_google_user(google_token: GoogleToken):
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
        logger.error("Synchronous enhancement called but upsampler is None.")
        raise RuntimeError("Upsampler not initialized for synchronous enhancement.")
    
    processed_cv = None
    try:
        if not image_cv_data.flags['C_CONTIGUOUS']:
            logger.debug("Input image_cv_data is not C-contiguous. Converting.")
            image_cv_data = np.ascontiguousarray(image_cv_data)

        logger.debug(f"Starting RealESRGAN enhance. Input shape: {image_cv_data.shape}, outscale: {output_scale_factor}")
        processed_cv, _ = current_upsampler.enhance(image_cv_data, outscale=output_scale_factor)
        logger.debug(f"RealESRGAN enhance finished. Output shape: {processed_cv.shape if processed_cv is not None else 'None'}")
        return processed_cv
    except Exception as e:
        logger.error(f"Error during synchronous_enhancement_function: {e}", exc_info=True)
        raise 
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Called torch.cuda.empty_cache() in synchronous_enhancement_function.")

@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    if upsampler is None:
        logger.error(f"Upscaling attempt by {current_user.email} but upsampler not available (was None at request time).")
        raise HTTPException(status_code=503, detail="Image upscaling service is currently unavailable or not initialized.")

    logger.info(f"Processing image request for user: {current_user.email}, file: {file.filename}")

    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded by {current_user.email}: {file.content_type}")
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image type.")

    contents = None
    pil_image_processing = None 
    img_cv = None            
    processed_image_cv = None

    try:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE_BYTES:
            logger.warning(f"File size limit exceeded: {len(contents)} bytes for {file.filename}")
            raise HTTPException(status_code=413, detail=f"Image file size exceeds limit of {MAX_FILE_SIZE_MB}MB.")

        original_pil_format_name = "PNG" 

        try: # Image Pre-processing Block
            pil_image_processing = Image.open(io.BytesIO(contents))
            original_pil_format_name = pil_image_processing.format or "PNG"
            original_width, original_height = pil_image_processing.size
            current_pixels = original_width * original_height
            logger.info(f"Original: {original_width}x{original_height} ({current_pixels}px) for {file.filename}")

            # --- Step 1: Downsize if EXCEEDING MAX_IMAGE_PIXELS (before ESRGAN) ---
            if current_pixels > MAX_IMAGE_PIXELS:
                logger.info(f"Original ({current_pixels}px) exceeds MAX_IMAGE_PIXELS ({MAX_IMAGE_PIXELS}). Downsizing input for ESRGAN.")
                target_pixels_for_resize = MAX_IMAGE_PIXELS -1 if MAX_IMAGE_PIXELS > 0 else 1
                scale_ratio = sqrt(target_pixels_for_resize / current_pixels)
                new_width = max(1, int(original_width * scale_ratio))
                new_height = max(1, int(original_height * scale_ratio))
                pil_image_processing = pil_image_processing.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Input for ESRGAN downsized to: {pil_image_processing.width}x{pil_image_processing.height} ({pil_image_processing.width * pil_image_processing.height}px)")
            
            current_w, current_h = pil_image_processing.size

            # --- Step 2: PRE-UPSCALING IF NEEDED for RealESRGAN input ---
            if NETSCALE <= 0: 
                logger.error("NETSCALE (output scale factor) must be positive.")
            else:
                projected_esrgan_output_w = current_w * NETSCALE 
                projected_esrgan_output_h = current_h * NETSCALE
                
                min_input_w_for_final_target = ceil(TARGET_FINAL_WIDTH / NETSCALE)
                min_input_h_for_final_target = ceil(TARGET_FINAL_HEIGHT / NETSCALE)

                if current_w < min_input_w_for_final_target or current_h < min_input_h_for_final_target:
                    logger.info(f"Image ({current_w}x{current_h}) when scaled by NETSCALE={NETSCALE} "
                                f"to ({projected_esrgan_output_w}x{projected_esrgan_output_h}) "
                                f"might be too small relative to final target ({TARGET_FINAL_WIDTH}x{TARGET_FINAL_HEIGHT}). Pre-upscaling input for RealESRGAN.")
                    
                    new_pre_w = current_w
                    new_pre_h = current_h
                    
                    if current_w < min_input_w_for_final_target:
                        new_pre_w = min_input_w_for_final_target
                    if current_h < min_input_h_for_final_target:
                        new_pre_h = min_input_h_for_final_target
                    
                    if current_w > 0 and current_h > 0: # Maintain aspect ratio for this pre-upscale
                        original_aspect = current_w / current_h
                        if new_pre_w / original_aspect > new_pre_h: 
                            new_pre_h = ceil(new_pre_w / original_aspect)
                        else: 
                            new_pre_w = ceil(new_pre_h * original_aspect)

                    if new_pre_w > current_w or new_pre_h > current_h:
                        pil_image_processing = pil_image_processing.resize((new_pre_w, new_pre_h), Image.Resampling.LANCZOS)
                        logger.info(f"Pre-upscaled input for RealESRGAN to: {pil_image_processing.width}x{pil_image_processing.height}")
                        # current_w, current_h = pil_image_processing.size # Not needed as these are not used further in this block
                    else:
                        logger.info("No pre-upscale for RealESRGAN input needed based on final target.")
            
            # Convert PIL Image to OpenCV format for RealESRGAN
            if pil_image_processing.mode == 'RGBA': img_cv = cv2.cvtColor(np.array(pil_image_processing), cv2.COLOR_RGBA2BGRA)
            elif pil_image_processing.mode == 'RGB': img_cv = cv2.cvtColor(np.array(pil_image_processing), cv2.COLOR_RGB2BGR)
            elif pil_image_processing.mode == 'L': img_cv = cv2.cvtColor(np.array(pil_image_processing), cv2.COLOR_GRAY2BGR)
            elif pil_image_processing.mode == 'P': 
                logger.info("Converting PIL image from P (Palette) mode to RGBA then BGRA for OpenCV.")
                pil_image_processing_converted = pil_image_processing.convert('RGBA')
                img_cv = cv2.cvtColor(np.array(pil_image_processing_converted), cv2.COLOR_RGBA2BGRA)
            else: 
                logger.info(f"Converting PIL image from mode {pil_image_processing.mode} to RGB then BGR for OpenCV.")
                pil_converted = pil_image_processing.convert('RGB')
                img_cv = cv2.cvtColor(np.array(pil_converted), cv2.COLOR_RGB2BGR)
        
        except UnidentifiedImageError:
            logger.error(f"Pillow UnidentifiedImageError for {file.filename}", exc_info=True)
            raise HTTPException(status_code=400, detail="Cannot identify image file.")
        except Exception as e_prep:
            logger.error(f"Error preparing image {file.filename}: {e_prep}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Error preparing image: {str(e_prep)}")
        
        if img_cv is None: 
            logger.error(f"img_cv is None after preparation steps for {file.filename}")
            raise HTTPException(status_code=500, detail="Image preparation failed to produce OpenCV image.")

        logger.info(f"Input for RealESRGAN (img_cv shape): {img_cv.shape}")

        # --- Step 3: RealESRGAN ENHANCEMENT ---
        loop = asyncio.get_event_loop()
        enhance_task_with_args = partial(synchronous_enhancement_function, img_cv, upsampler, NETSCALE)
        
        processed_image_cv = await asyncio.wait_for(
            loop.run_in_executor(thread_pool_executor, enhance_task_with_args),
            timeout=ENHANCE_TIMEOUT_SECONDS
        )
        
        gc.collect()
        logger.debug("Called gc.collect() in process_image after threaded task.")

        if processed_image_cv is None:
            logger.error(f"RealESRGAN enhancement returned None for {file.filename}")
            raise HTTPException(status_code=500, detail="Image enhancement failed.")

        logger.info(f"Enhancement output shape: {processed_image_cv.shape}")

        # Convert enhanced OpenCV image back to PIL
        processed_pil_image = None
        if len(processed_image_cv.shape) == 3 and processed_image_cv.shape[2] == 4: # BGRA
            processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image_cv, cv2.COLOR_BGRA2RGBA))
        elif len(processed_image_cv.shape) == 3 and processed_image_cv.shape[2] == 3: # BGR
            processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2RGB))
        else: 
             logger.error(f"Unexpected RealESRGAN output shape: {processed_image_cv.shape}")
             raise HTTPException(status_code=500, detail="Image processing failed: unexpected enhancer output format.")

        # --- Step 4: RESIZE TO COVER FINAL TARGET AND CROP TO FIT EXACTLY ---
        logger.info(f"Image after ESRGAN, before final resize & crop: {processed_pil_image.width}x{processed_pil_image.height}")
        
        image_to_save = processed_pil_image # Default in case of issues

        try:
            esrgan_w, esrgan_h = processed_pil_image.size
            target_w = TARGET_FINAL_WIDTH
            target_h = TARGET_FINAL_HEIGHT

            if esrgan_w == 0 or esrgan_h == 0:
                logger.warning("ESRGAN output image has zero width or height. Cannot perform resize/crop to target.")
            else:
                img_aspect = esrgan_w / esrgan_h
                target_aspect = target_w / target_h

                interim_w, interim_h = esrgan_w, esrgan_h # Start with current dims

                if img_aspect >= target_aspect: # Image is wider or same aspect as target
                    # Scale by height to match target_h. Width will be >= target_w.
                    scale_factor = target_h / esrgan_h
                    interim_h = target_h
                    interim_w = int(round(esrgan_w * scale_factor)) # Use round for better precision
                else: # Image is taller than target
                    # Scale by width to match target_w. Height will be >= target_h.
                    scale_factor = target_w / esrgan_w
                    interim_w = target_w
                    interim_h = int(round(esrgan_h * scale_factor)) # Use round

                # Ensure interim dimensions are at least target dimensions (safety for rounding)
                # If after scaling, one dimension is slightly less than target due to rounding, force it.
                # This is more robust if the calculated interim_w/h is, e.g., target_w-1.
                if interim_w < target_w: interim_w = target_w
                if interim_h < target_h: interim_h = target_h
                
                logger.info(f"Resizing ESRGAN output from {esrgan_w}x{esrgan_h} to interim {interim_w}x{interim_h} (to cover {target_w}x{target_h})")
                resized_image = processed_pil_image.resize((interim_w, interim_h), Image.Resampling.LANCZOS)

                # Center crop to target dimensions
                left = (interim_w - target_w) / 2
                top = (interim_h - target_h) / 2
                # For crop, we want box of size target_w, target_h
                # So, right = left + target_w and bottom = top + target_h
                right = left + target_w 
                bottom = top + target_h
                
                crop_box = (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))

                logger.info(f"Cropping interim image at {crop_box} to final {target_w}x{target_h}")
                image_to_save = resized_image.crop(crop_box)

                # Final check and forceful resize if crop wasn't exact (e.g. off by 1px due to rounding)
                if image_to_save.width != target_w or image_to_save.height != target_h:
                    logger.warning(f"Cropped image is {image_to_save.width}x{image_to_save.height}, not exactly {target_w}x{target_h}. "
                                   "Performing a final exact resize. This might introduce slight distortion if crop was significantly off.")
                    image_to_save = image_to_save.resize((target_w, target_h), Image.Resampling.LANCZOS)

        except Exception as e_resize_crop:
            logger.error(f"Error during final resize/crop to target dimensions: {e_resize_crop}", exc_info=True)
            logger.warning("Using un-modified image (from ESRGAN) due to final resize/crop error.")
            image_to_save = processed_pil_image 

        logger.info(f"Final output dimensions after resize & crop: {image_to_save.width}x{image_to_save.height}")

        # --- Step 5: PREPARE RESPONSE ---
        img_byte_arr = io.BytesIO()
        output_format_str = original_pil_format_name.upper() 

        if output_format_str == "JPEG" and image_to_save.mode in ['RGBA', 'LA', 'P']:
            logger.info(f"Output format is JPEG and image mode is {image_to_save.mode}. Converting to RGB.")
            image_to_save = image_to_save.convert('RGB')
        
        if output_format_str == "GIF" and hasattr(image_to_save, 'is_animated') and image_to_save.is_animated:
             logger.info("Saving as animated GIF.")
             image_to_save.save(img_byte_arr, format=output_format_str, save_all=True, 
                                duration=image_to_save.info.get('duration', 100), 
                                loop=image_to_save.info.get('loop', 0))
        else:
            save_params = {}
            if output_format_str == "JPEG":
                save_params['quality'] = 90 
            elif output_format_str == "PNG":
                save_params['optimize'] = True
            
            image_to_save.save(img_byte_arr, format=output_format_str, **save_params)
        
        img_byte_arr.seek(0)
        
        media_type_suffix = output_format_str.lower()
        if media_type_suffix == "jpeg": media_type_suffix = "jpg" 
        media_type = f"image/{media_type_suffix}"

        logger.info(f"Successfully processed {file.filename}. Returning as {media_type}.")
        return StreamingResponse(img_byte_arr, media_type=media_type)

    except asyncio.TimeoutError:
        logger.error(f"Processing timed out after {ENHANCE_TIMEOUT_SECONDS}s for {file.filename}", exc_info=True)
        raise HTTPException(status_code=504, detail=f"Processing timed out after {ENHANCE_TIMEOUT_SECONDS} seconds.")
    except HTTPException: 
        raise
    except cv2.error as e_cv:
        logger.error(f"OpenCV error for {file.filename}: {e_cv}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OpenCV processing error: {str(e_cv)}")
    except Exception as e_main: 
        logger.error(f"Generic error processing image {file.filename}: {e_main}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Called torch.cuda.empty_cache() in generic exception handler of process_image.")
        gc.collect()
        logger.debug("Called gc.collect() in generic exception handler of process_image.")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during image processing: {str(e_main)}")
    finally:
        if contents: del contents 
        if pil_image_processing: del pil_image_processing
        if img_cv is not None: del img_cv 
        if processed_image_cv is not None: del processed_image_cv 
        logger.debug("process_image final cleanup: Deleted local large variables.")


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)): return current_user

if __name__ == "__main__":
    import uvicorn
    
    if not os.path.exists('weights'):
        os.makedirs('weights', exist_ok=True)
        logger.info("Created 'weights' directory.")

    if not os.path.exists('weights/RealESRGAN_x4plus.pth'):
        logger.warning("RealESRGAN_x4plus.pth not found in 'weights' directory. Upscaling will fail.")
        print("\n--- IMPORTANT ---")
        print("RealESRGAN model weights not found.")
        print("Please download 'RealESRGAN_x4plus.pth' from the official Real-ESRGAN repository")
        print("e.g., https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
        print("and place it in the 'weights' directory next to this script.")
        print("-----------------\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)