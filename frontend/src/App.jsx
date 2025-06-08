import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleLogin, googleLogout } from '@react-oauth/google';
import jwt_decode from 'jwt-decode';
import { Oval } from 'react-loader-spinner';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import './App.css'; // Ensure ReactCrop.css is imported here or in main.jsx via App.css

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8181'; // Or your production URL

// Helper function to get cropped image data (This is from your original file, unchanged)
function getCroppedImg(image, crop, fileName) {
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
      resolve(blob);
    }, 'image/png', 0.95); 
  });
}


function App() {
  // Page State (from original)
  const [currentPage, setCurrentPage] = useState('upload'); 

  // Auth State (from original)
  const [user, setUser] = useState(null); // Will now store the full user object { email, name, tier }
  const [appToken, setAppToken] = useState(localStorage.getItem('appToken'));

  // Image & Crop State (from original)
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImagePreview, setOriginalImagePreview] = useState(null);
  const [crop, setCrop] = useState();
  const [completedCrop, setCompletedCrop] = useState(null);
  const [croppedImageBlob, setCroppedImageBlob] = useState(null);
  const [croppedImagePreviewUrl, setCroppedImagePreviewUrl] = useState(null); 
  
  const imgRef = useRef(null);
  const aspect = 16 / 9;

  // Processing & Result State (from original)
  const [processedImageUrl, setProcessedImageUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [downloadFileName, setDownloadFileName] = useState('frame_tv_art.png');
  const processedImageRef = useRef(null); 
  const fileInputRef = useRef(null);

  // resetAllImageStates (from original, unchanged)
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
        fileInputRef.current.value = "";
    }
    setCurrentPage('upload');
    if (clearError) setError('');
  }, [originalImagePreview, processedImageUrl, croppedImagePreviewUrl]); 

  // useEffect for initial auth check (from original, with one change)
  useEffect(() => { 
    const token = localStorage.getItem('appToken'); 
    if (token) {
      try {
        const decodedToken = jwt_decode(token);
        if (decodedToken.exp * 1000 < Date.now()) {
          handleLogout(); 
        } else {
           // --- CHANGE #1 (START) ---
           // Instead of decoding user info from the token, fetch the full, authoritative user object
           // from the backend. This ensures we always have the latest tier information.
           setAppToken(token); 
           fetch(`${API_BASE_URL}/users/me`, { headers: { 'Authorization': `Bearer ${token}` }})
            .then(res => {
                if (res.ok) return res.json();
                // If the token is rejected by the backend (e.g., 401), log out.
                throw new Error('Token validation failed on backend');
            })
            .then(userDataFromServer => {
                setUser(userDataFromServer); // Set the full user object
            })
            .catch(err => {
                console.error("Failed to re-validate session with /users/me:", err);
                handleLogout(); // Log out if we can't get user data
            });
           // --- CHANGE #1 (END) ---
        }
      } catch (e) {
        console.error("Error decoding app token on load:", e);
        handleLogout();
      }
    }
  }, []); // Run only once on mount 

  // handleGoogleLoginSuccess (from original, with one change)
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
        setUser(data.user); // The backend now provides the full user object with the tier
        setCurrentPage('upload'); 
        setError('');
    } catch (err) { 
        console.error('Backend Google login failed:', err); 
        setError(err.message); 
        handleLogout(); 
    } 
    finally { setIsLoading(false); }
  };
  const handleGoogleLoginError = () => {  /* Unchanged */
      console.error('Google Login Failed on Frontend');
      setError('Google login failed. Please try again.');
  };

  // handleLogout (from original, unchanged)
  const handleLogout = () => { 
    console.log("Logging out...");
    googleLogout(); 
    localStorage.removeItem('appToken');
    setAppToken(null); 
    setUser(null);
    resetAllImageStates(true); 
  };

  // useEffect for revoking object URLs (from original, unchanged)
  useEffect(() => { 
    const urlsToRevoke = [originalImagePreview, processedImageUrl, croppedImagePreviewUrl];
    return () => { urlsToRevoke.forEach(url => { if (url) URL.revokeObjectURL(url); }); };
  }, [originalImagePreview, processedImageUrl, croppedImagePreviewUrl]);

  // onImageLoad (from original, unchanged)
  const onImageLoad = useCallback((e) => { 
    imgRef.current = e.currentTarget;
    const { width, height } = e.currentTarget;
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
    setCrop(centeredCrop); 
    setCompletedCrop(centeredCrop); 
    return false;
  }, [aspect, originalImagePreview, setCurrentPage, setError]);

  // handleFileChange (from original, unchanged)
  const handleFileChange = (event) => {
    const file = event.target.files && event.target.files[0]; 
    if (processedImageUrl) { URL.revokeObjectURL(processedImageUrl); setProcessedImageUrl(null); }
    if (croppedImagePreviewUrl) { URL.revokeObjectURL(croppedImagePreviewUrl); setCroppedImagePreviewUrl(null); }
    setCroppedImageBlob(null);
    if (originalImagePreview) { URL.revokeObjectURL(originalImagePreview); } 
    setOriginalImagePreview(null); 
    setCrop(undefined); 
    setCompletedCrop(null);
    setError(''); 
    if (file) {
      setSelectedFile(file); 
      setOriginalImagePreview(URL.createObjectURL(file));
      setCurrentPage('crop');
    } else {
      setSelectedFile(null); 
      setCurrentPage('upload'); 
    }
  };
  
  // useEffect for crop preview (from original, unchanged)
  useEffect(() => { 
    if (completedCrop?.width && completedCrop?.height && imgRef.current && selectedFile) {
      generateCroppedPreview(imgRef.current, completedCrop);
    }
  }, [completedCrop, selectedFile]); 

  // generateCroppedPreview (from original, unchanged)
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

  // handleProcessCroppedImage (from original, with changes for tier logic)
  const handleProcessCroppedImage = async () => { 
    if (!croppedImageBlob) { setError('Please make a crop selection and ensure a preview is visible.'); return; }
    if (!appToken) { setError('Please sign in.'); return; }
    
    setIsLoading(true); 
    setError('');
    
    // --- CHANGE #3 (START) ---
    // Check the user's tier to determine the endpoint and final filename.
    const isPremium = user?.tier === 'premium';
    const endpoint = isPremium ? `${API_BASE_URL}/process-image-premium/` : `${API_BASE_URL}/process-image/`;
    console.log(`User tier: ${user?.tier}. Using endpoint: ${endpoint}`);
    // --- CHANGE #3 (END) ---

    const formData = new FormData();
    const fName = croppedImageBlob.name || 'cropped_for_tv.png';
    formData.append('file', croppedImageBlob, fName);
    
    // Original AbortController logic is good, keeping it
    const controller = new AbortController();
    const tId = setTimeout(() => controller.abort(), 600000); // 10 minutes

    try {
        const response = await fetch(endpoint, {method: 'POST', headers: {'Authorization': `Bearer ${appToken}`}, body: formData, signal: controller.signal});
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
        
        // --- CHANGE #4 (START) ---
        // Dynamically set the download filename
        const bn = fName.replace(/\.[^/.]+$/, "")||'art'; 
        const fileExtension = response.headers.get('content-type')?.includes('jpeg') ? 'jpg' : 'png';
        setDownloadFileName(`${isPremium ? 'premium-enhanced' : 'resized'}_${bn}.${fileExtension}`);
        // --- CHANGE #4 (END) ---

        setCurrentPage('result');
    } catch (err) { 
        clearTimeout(tId); 
        if(err.name==='AbortError'){setError('Image enhancement timed out. Try a smaller selection or image.');}
        else{setError(err.message||'Image processing failed.');} 
        setProcessedImageUrl(null);
    }
    finally { setIsLoading(false); }
  };

  // handleDownload (from original, unchanged)
  const handleDownload = () => { 
    if (processedImageRef.current) {
      const url = URL.createObjectURL(processedImageRef.current); 
      const a = document.createElement('a'); a.href = url; a.download = downloadFileName;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url); 
    }
  };

  // renderPageContent (from original, with minor text changes for tier logic)
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
          <GoogleLogin onSuccess={handleGoogleLoginSuccess} onError={handleGoogleLoginError} theme="outline" size="large" shape="rectangular"/>
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
            <p>Choose any image you'd like to prepare for your Samsung Frame TV's Art Mode.</p>
            <input type="file" accept="image/*" onChange={handleFileChange} ref={fileInputRef}/>
          </div>
        );
      case 'crop':
        return originalImagePreview ? (
          <div className="crop-area-container">
            <h2>Step 2: Crop for 16:9 Aspect</h2>
            <p>Drag to select the perfect 16:9 portion of your image.</p> 
            <ReactCrop crop={crop} onChange={(_, pc) => setCrop(pc)} onComplete={(c) => setCompletedCrop(c)} aspect={aspect} minWidth={50} minHeight={Math.round(50/aspect)}>
              <img ref={imgRef} alt="Crop area" src={originalImagePreview} onLoad={onImageLoad} style={{maxHeight: '350px', maxWidth: '100%', display: 'block', margin: '0 auto', objectFit: 'contain'}}/>
            </ReactCrop>
            <div className="crop-actions-container">
              {croppedImagePreviewUrl && (
                <div className="crop-output-preview-container">
                  <h4>Cropped Preview:</h4>
                  <img alt="Cropped Preview" src={croppedImagePreviewUrl} className="crop-output-image"/>
                </div>
              )}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                <button onClick={handleProcessCroppedImage} disabled={!croppedImageBlob || isLoading} className="action-button">
                  {/* Text dynamically changes based on tier */}
                  {user?.tier === 'premium' ? 'Enhance with AI (Premium)' : 'Resize to 4K (Free)'}
                </button>
                <button onClick={() => resetAllImageStates(true)} className="secondary-action-button">
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
                <button onClick={() => resetAllImageStates(true)} className="secondary-action-button">
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
          {/* This part of the UI is now driven by the 'user' object from the backend */}
          {appToken && user && !isLoading && ( 
            <div className="user-info">
              {/* --- CHANGE #2 (START) --- */}
              {user.tier === 'premium' && <span title="Premium Tier" style={{color: '#ffd700', fontWeight: 'bold', fontSize: '1.2rem'}}>👑</span>}
              {/* --- CHANGE #2 (END) --- */}
              {user.picture && <img src={user.picture} alt={user.name || 'User'} className="user-avatar" />}
              <span>Hi, {user.name || user.email}!</span>
              <button onClick={handleLogout} className="logout-button">Logout</button>
            </div>
          )}
        </header>
        
        <div className={!appToken && !isLoading ? "auth-section" : "main-content-area"}>
          {renderPageContent()}
        </div>

        <div className="disclaimer">
          {/* Unchanged */}
        </div>
      </div>
    </div>
  );
}

export default App;