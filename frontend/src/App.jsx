import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleLogin, googleLogout } from '@react-oauth/google';
import jwt_decode from 'jwt-decode';
import { Oval } from 'react-loader-spinner';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import './App.css'; // Ensure ReactCrop.css is imported here or in main.jsx via App.css

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8181'; // Or your production URL

// Helper function to get cropped image data
function getCroppedImg(image, crop, fileName) {
  // console.log("getCroppedImg called with crop:", crop); // Debug
  if (!crop || !image || crop.width === 0 || crop.height === 0) {
    console.error("getCroppedImg: Invalid crop or image dimensions.");
    return Promise.reject(new Error("Invalid crop or image dimensions."));
  }

  const canvas = document.createElement('canvas');
  const scaleX = image.naturalWidth / image.width;
  const scaleY = image.naturalHeight / image.height;

  const canvasWidth = Math.floor(crop.width * scaleX);
  const canvasHeight = Math.floor(crop.height * scaleY);

  if (canvasWidth === 0 || canvasHeight === 0) {
    console.error("getCroppedImg: Calculated canvas dimensions are zero.");
    return Promise.reject(new Error("Calculated canvas dimensions are zero."));
  }

  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    console.error("getCroppedImg: Could not get 2D context from canvas.");
    return Promise.reject(new Error("Could not get 2D context."));
  }
  
  ctx.imageSmoothingQuality = 'high'; 

  ctx.drawImage(
    image,
    crop.x * scaleX,
    crop.y * scaleY,
    crop.width * scaleX,
    crop.height * scaleY,
    0,
    0,
    canvasWidth, 
    canvasHeight
  );

  return new Promise((resolve, reject) => {
    canvas.toBlob(blob => {
      if (!blob) {
        console.error('Canvas toBlob resulted in null blob.');
        reject(new Error('Could not create blob from canvas.'));
        return;
      }
      blob.name = fileName; 
      // console.log("Cropped blob created:", blob); // Debug
      resolve(blob);
    }, 'image/png', 0.95); 
  });
}


function App() {
  // Page State
  const [currentPage, setCurrentPage] = useState('upload'); // 'upload', 'crop', 'result'

  // Auth State
  const [user, setUser] = useState(null);
  const [appToken, setAppToken] = useState(localStorage.getItem('appToken'));

  // Image & Crop State
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImagePreview, setOriginalImagePreview] = useState(null);
  const [crop, setCrop] = useState();
  const [completedCrop, setCompletedCrop] = useState(null);
  const [croppedImageBlob, setCroppedImageBlob] = useState(null);
  const [croppedImagePreviewUrl, setCroppedImagePreviewUrl] = useState(null); 
  
  const imgRef = useRef(null);
  const aspect = 16 / 9;

  // Processing & Result State
  const [processedImageUrl, setProcessedImageUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [downloadFileName, setDownloadFileName] = useState('frame_tv_art.png');
  const processedImageRef = useRef(null); 
  const fileInputRef = useRef(null);


  const resetAllImageStates = useCallback((clearError = true) => {
    console.log("resetAllImageStates called");
    setSelectedFile(null);
    if(originalImagePreview) URL.revokeObjectURL(originalImagePreview);
    setOriginalImagePreview(null);
    if(processedImageUrl) URL.revokeObjectURL(processedImageUrl);
    setProcessedImageUrl(null);
    if(croppedImagePreviewUrl) URL.revokeObjectURL(croppedImagePreviewUrl);
    setCroppedImagePreviewUrl(null);
    setCroppedImageBlob(null);
    setCrop(undefined);
    setCompletedCrop(null);
    processedImageRef.current = null;
    if (fileInputRef.current) {
        console.log("Clearing file input value via ref in resetAllImageStates.");
        fileInputRef.current.value = "";
    }
    setCurrentPage('upload');
    if (clearError) setError('');
  }, [originalImagePreview, processedImageUrl, croppedImagePreviewUrl]); 

  useEffect(() => { 
    const token = localStorage.getItem('appToken'); 
    if (token) {
      try {
        const decodedAppToken = jwt_decode(token);
        if (decodedAppToken.exp * 1000 < Date.now()) {
          handleLogout(); 
        } else {
           if (decodedAppToken.user_info) {
             setUser(decodedAppToken.user_info);
           }
           setAppToken(token); 
        }
      } catch (e) {
        console.error("Error decoding app token on load:", e);
        handleLogout();
      }
    }
  }, []); // Run only once on mount 

  const handleGoogleLoginSuccess = async (credentialResponse) => { 
    const idToken = credentialResponse.credential;
    setIsLoading(true); 
    setError('');
    try {
        const response = await fetch(`${API_BASE_URL}/auth/google`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({token: idToken}),});
        if (!response.ok) {const errData = await response.json().catch(() => ({detail: "Google login failed."})); throw new Error(errData.detail || `Google login error ${response.status}`);}
        const data = await response.json();
        localStorage.setItem('appToken', data.access_token);
        setAppToken(data.access_token); 
        setUser(data.user); 
        setCurrentPage('upload'); 
        setError(''); // Clear any previous login errors
    } catch (err) { 
        console.error('Backend Google login failed:', err); 
        setError(err.message); 
        handleLogout(); 
    } 
    finally { setIsLoading(false); }
  };
  const handleGoogleLoginError = () => { 
      console.error('Google Login Failed on Frontend');
      setError('Google login failed. Please try again.');
  };

  // Use useCallback for handleLogout if it were passed as a prop or in a dependency array,
  // but here it's fine as a regular const if only called directly.
  const handleLogout = () => { 
    console.log("Logging out...");
    googleLogout(); 
    localStorage.removeItem('appToken');
    setAppToken(null); 
    setUser(null);
    resetAllImageStates(true); 
  };

  useEffect(() => { 
    const urlsToRevoke = [originalImagePreview, processedImageUrl, croppedImagePreviewUrl];
    return () => { urlsToRevoke.forEach(url => { if (url) URL.revokeObjectURL(url); }); };
  }, [originalImagePreview, processedImageUrl, croppedImagePreviewUrl]);

  const onImageLoad = useCallback((e) => { 
    console.log("onImageLoad triggered for ReactCrop"); 
    imgRef.current = e.currentTarget;
    const { width, height } = e.currentTarget;
    console.log("Image dimensions for crop init:", width, height);

    if (width === 0 || height === 0) { 
        console.error("onImageLoad: Image loaded with zero dimensions.");
        setError("Could not load image. Please try a different image or check the file.");
        if(originalImagePreview) URL.revokeObjectURL(originalImagePreview);
        setOriginalImagePreview(null); 
        setCurrentPage('upload'); 
        return false; 
    }
    const newCrop = makeAspectCrop({ unit: '%', width: 100, }, aspect, width, height);
    const centeredCrop = centerCrop(newCrop, width, height);
    console.log("Setting initial crop in onImageLoad:", JSON.stringify(centeredCrop));
    setCrop(centeredCrop); 
    setCompletedCrop(centeredCrop); 
    return false;
  }, [aspect, originalImagePreview, setCurrentPage, setError]); // Added dependencies

  const handleFileChange = (event) => {
    console.log("handleFileChange triggered. Initial event.target.files:", event.target.files);
    const file = event.target.files && event.target.files[0]; 

    // Clear previous processing/crop results before setting new file
    if (processedImageUrl) { URL.revokeObjectURL(processedImageUrl); setProcessedImageUrl(null); }
    if (croppedImagePreviewUrl) { URL.revokeObjectURL(croppedImagePreviewUrl); setCroppedImagePreviewUrl(null); }
    setCroppedImageBlob(null);
    if (originalImagePreview) { URL.revokeObjectURL(originalImagePreview); } 
    setOriginalImagePreview(null); 
    setCrop(undefined); 
    setCompletedCrop(null);
    setError(''); 

    if (file) {
      console.log("File actually selected:", file.name);
      setSelectedFile(file); 
      
      const newObjectUrl = URL.createObjectURL(file);
      console.log("Setting new originalImagePreview URL:", newObjectUrl.substring(0, 50) + "...");
      setOriginalImagePreview(newObjectUrl);
      setCurrentPage('crop');
    } else {
      console.log("No file was selected (e.g., user cancelled dialog).");
      setSelectedFile(null); 
      // originalImagePreview is already null from above
      setCurrentPage('upload'); 
    }
  };
  
  useEffect(() => { 
    if (completedCrop?.width && completedCrop?.height && imgRef.current && selectedFile) {
      generateCroppedPreview(imgRef.current, completedCrop);
    }
  }, [completedCrop, selectedFile]); 

  async function generateCroppedPreview(image, cropData) { 
    if (!cropData || !image) return;
    const fileName = selectedFile ? selectedFile.name : 'crop.png';
    try {
      const blob = await getCroppedImg(image, cropData, fileName);
      setCroppedImageBlob(blob);
      if(croppedImagePreviewUrl) URL.revokeObjectURL(croppedImagePreviewUrl);
      setCroppedImagePreviewUrl(URL.createObjectURL(blob));
    } catch (e) { 
        console.error("Error in generateCroppedPreview:", e); 
        setError("Crop preview failed. Please adjust crop or try a different image."); 
        setCroppedImageBlob(null); 
        setCroppedImagePreviewUrl(null);
    }
  }

  const handleProcessCroppedImage = async () => { 
    if (!croppedImageBlob) { setError('Please make a crop selection and ensure a preview is visible.'); return; }
    if (!appToken) { setError('Please sign in.'); return; }
    
    setIsLoading(true); 
    setError('');
    
    const formData = new FormData();
    const fName = croppedImageBlob.name || 'cropped_for_tv.png';
    formData.append('file', croppedImageBlob, fName);
    const controller = new AbortController();
    const tId = setTimeout(() => controller.abort(), 600000); // 10 minutes
    try {
        const response = await fetch(`${API_BASE_URL}/process-image/`, {method: 'POST', headers: {'Authorization': `Bearer ${appToken}`}, body: formData, signal: controller.signal});
        clearTimeout(tId);
        if (!response.ok) { 
            let errDetail = `Error ${response.status}: ${response.statusText}`; 
            if (response.status === 401) { errDetail = "Unauthorized. Session may have expired."; handleLogout(); } 
            else { try {const ed = await response.json(); errDetail = ed.detail || errDetail;} catch (e) { /* use default errDetail */ } }
            throw new Error(errDetail);
        }
        const imgBlob = await response.blob();
        processedImageRef.current = imgBlob; 
        setProcessedImageUrl(URL.createObjectURL(imgBlob));
        
        let ext = 'png'; 
        const ct = response.headers.get('content-type'); 
        if(ct?.startsWith('image/')) {const tp = ct.split('/')[1].toLowerCase(); if(tp==='jpeg'||tp==='jpg') ext='jpg'; else if(tp==='png') ext='png';}
        const bn = fName.replace(/\.[^/.]+$/, "")||'art'; 
        setDownloadFileName(`enhanced_${bn}.${ext}`);
        setCurrentPage('result');
    } catch (err) { 
        clearTimeout(tId); 
        if(err.name==='AbortError'){setError('Image enhancement timed out. Try a smaller selection or image.');}
        else{setError(err.message||'Image processing failed.');} 
        setProcessedImageUrl(null);
    }
    finally { setIsLoading(false); }
  };

  const handleDownload = () => { 
    if (processedImageRef.current) {
      const url = URL.createObjectURL(processedImageRef.current); 
      const a = document.createElement('a'); a.href = url; a.download = downloadFileName;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url); 
    }
  };

  const renderPageContent = () => {
    if (isLoading) { 
      return (
        <div className="loading-indicator">
          <Oval height={50} width={50} color="#673ab7" secondaryColor="#d1c4e9" strokeWidth={4} strokeWidthSecondary={4} ariaLabel="oval-loading" wrapperStyle={{ margin: "0 auto" }} visible={true}/>
          <p>Enhancing your image for The Frame TV...</p>
        </div>
      );
    }

    if (!appToken) { 
      return ( 
        <>
          <p className="auth-intro-text">
            Sign in to upscale your favorite images for your Samsung The Frame TV.
          </p>
          {error && <p className="error" style={{marginBottom: '15px'}}>Error: {error}</p>}
          <GoogleLogin
            onSuccess={handleGoogleLoginSuccess}
            onError={handleGoogleLoginError}
            theme="outline" size="large" shape="rectangular"
          />
        </>
      );
    }
    
    if (error) { 
        return (
            <div>
                <p className="error">Error: {error}</p>
                <button onClick={() => resetAllImageStates(true) } className="action-button" style={{marginTop: "10px", backgroundColor: "#9575cd"}}>Start Over</button>
            </div>
        );
    }

    switch (currentPage) {
      case 'upload':
        return (
          <div className="upload-section">
            <h2>Step 1: Upload Your Image</h2>
            <p> 
              Choose any image you'd like to prepare for your Samsung Frame TV's Art Mode.
            </p>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleFileChange} 
              ref={fileInputRef}
            />
          </div>
        );
      case 'crop':
        return originalImagePreview ? (
          <div className="crop-area-container">
            <h2>Step 2: Crop for 16:9 Aspect</h2>
            <p>Drag to select the perfect 16:9 portion of your image.</p> 
            <ReactCrop 
              crop={crop} 
              onChange={(_, pc) => setCrop(pc)} 
              onComplete={(c) => setCompletedCrop(c)} 
              aspect={aspect} 
              minWidth={50} 
              minHeight={Math.round(50/aspect)}
            >
              <img 
                ref={imgRef} 
                alt="Crop area" 
                src={originalImagePreview} 
                onLoad={onImageLoad} 
                style={{maxHeight: '350px', maxWidth: '100%', display: 'block', margin: '0 auto', objectFit: 'contain'}}
              />
            </ReactCrop>
            <div className="crop-actions-container">
              {croppedImagePreviewUrl && (
                <div className="crop-output-preview-container">
                  <h4>Cropped Preview:</h4>
                  <img alt="Cropped Preview" src={croppedImagePreviewUrl} className="crop-output-image"/>
                </div>
              )}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                <button 
                  onClick={handleProcessCroppedImage} 
                  disabled={!croppedImageBlob || isLoading} 
                  className="action-button"
                >
                  Enhance Cropped Image
                </button>
                <button 
                  onClick={() => resetAllImageStates(true)} 
                  className="secondary-action-button"
                >
                  Choose Different Image
                </button>
              </div>
            </div>
          </div>
        ) : (
          <p>Please <button onClick={() => resetAllImageStates(true)} className="link-button">upload an image</button> first.</p>
        );
      case 'result':
        return processedImageUrl ? (
          <>
            <div className="image-preview-container" style={{marginTop: '10px'}}>
              <div className="image-box">
                <h2>Step 3: Your Frame TV Art!</h2>
                <img src={processedImageUrl} alt="Upscaled for Samsung Frame TV"/>
              </div>
            </div>
            <div className="result-section">
              <p>Save your upscaled image. <br/>Filename: <strong>{downloadFileName}</strong></p> 
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                <button onClick={handleDownload} className="download-button">Download Enhanced Image</button>
                <button 
                  onClick={() => resetAllImageStates(true)} 
                  className="secondary-action-button"
                >
                  Enhance Another Image
                </button>
              </div>
            </div>
          </>
        ) : (
            <p>Processing may have failed or no image is ready. Please <button onClick={() => resetAllImageStates(true)} className="link-button">start over</button>.</p>
        );
      default:
        return <p>Something went wrong. Please <button onClick={() => resetAllImageStates(true)} className="link-button">start over</button>.</p>;
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="app-header">
          <h1>Frame TV Image Enhancer</h1>
          {/* User info is displayed here when logged in & not loading */}
          {appToken && user && !isLoading && ( 
            <div className="user-info">
              {user.picture && <img src={user.picture} alt={user.name || 'User'} className="user-avatar" />}
              <span>Hi, {user.name || user.email}!</span>
              <button onClick={handleLogout} className="logout-button">Logout</button>
            </div>
          )}
        </header>
        
        {/* This div will conditionally get the "auth-section" class for the login prompt */}
        <div className={!appToken && !isLoading ? "auth-section" : "main-content-area"}> {/* Added a fallback class */}
          {renderPageContent()}
        </div>

        <div className="disclaimer">
          {/* ... disclaimer content ... */}
        </div>
      </div>
    </div>
  );
}

export default App;