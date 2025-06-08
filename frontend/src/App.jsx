import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleLogin, googleLogout } from '@react-oauth/google';
import jwt_decode from 'jwt-decode';
import { Oval } from 'react-loader-spinner';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import './App.css'; // Ensure ReactCrop.css is imported here or in main.jsx via App.css

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8181'; // Or your production URL

// Helper function to get cropped image data (FROM YOUR ORIGINAL, UNCHANGED)
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
  // --- All State from your original file ---
  const [currentPage, setCurrentPage] = useState('upload');
  const [user, setUser] = useState(null);
  const [appToken, setAppToken] = useState(localStorage.getItem('appToken'));
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImagePreview, setOriginalImagePreview] = useState(null);
  const [crop, setCrop] = useState();
  const [completedCrop, setCompletedCrop] = useState(null);
  const [croppedImageBlob, setCroppedImageBlob] = useState(null);
  const [croppedImagePreviewUrl, setCroppedImagePreviewUrl] = useState(null);
  const [processedImageUrl, setProcessedImageUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [downloadFileName, setDownloadFileName] = useState('frame_tv_art.png');
  // --- NEW STATE VARIABLE ADDED ---
  const [isProcessingAI, setIsProcessingAI] = useState(false);


  // --- All Refs from your original file (UNCHANGED) ---
  const imgRef = useRef(null);
  const aspect = 16 / 9;
  const processedImageRef = useRef(null);
  const fileInputRef = useRef(null);


  // --- All Functions from your original file (with necessary additions) ---

  const resetAllImageStates = useCallback((clearError = true) => { // UNCHANGED
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

  useEffect(() => { // MODIFIED
    const token = localStorage.getItem('appToken'); 
    if (token) {
      try {
        const decodedToken = jwt_decode(token);
        if (decodedToken.exp * 1000 < Date.now()) {
          handleLogout(); 
        } else {
           setAppToken(token); 
           fetch(`${API_BASE_URL}/users/me`, { headers: { 'Authorization': `Bearer ${token}` }})
            .then(res => {
                if (res.ok) return res.json();
                throw new Error('Token validation failed on backend');
            })
            .then(userDataFromServer => {
                setUser(userDataFromServer);
            })
            .catch(err => {
                console.error("Failed to re-validate session with /users/me:", err);
                handleLogout();
            });
        }
      } catch (e) {
        console.error("Error decoding app token on load:", e);
        handleLogout();
      }
    }
  }, []);

  const handleGoogleLoginSuccess = async (credentialResponse) => { // MODIFIED
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
        setError('');
    } catch (err) { 
        console.error('Backend Google login failed:', err); 
        setError(err.message); 
        handleLogout(); 
    } 
    finally { setIsLoading(false); }
  };
  const handleGoogleLoginError = () => {  // UNCHANGED
      console.error('Google Login Failed on Frontend');
      setError('Google login failed. Please try again.');
  };

  const handleLogout = () => { // UNCHANGED
    console.log("Logging out...");
    googleLogout(); 
    localStorage.removeItem('appToken');
    setAppToken(null); 
    setUser(null);
    resetAllImageStates(true); 
  };

  useEffect(() => { // UNCHANGED
    const urlsToRevoke = [originalImagePreview, processedImageUrl, croppedImagePreviewUrl];
    return () => { urlsToRevoke.forEach(url => { if (url) URL.revokeObjectURL(url); }); };
  }, [originalImagePreview, processedImageUrl, croppedImagePreviewUrl]);

  const onImageLoad = useCallback((e) => { // UNCHANGED
    imgRef.current = e.currentTarget;
    const { width, height } = e.currentTarget;
    const newCrop = makeAspectCrop({ unit: '%', width: 100, }, aspect, width, height);
    const centeredCrop = centerCrop(newCrop, width, height);
    setCrop(centeredCrop); 
    setCompletedCrop(centeredCrop); 
    return false;
  }, [aspect]);

  const handleFileChange = (event) => { // UNCHANGED
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
  
  useEffect(() => { // UNCHANGED
    if (completedCrop?.width && completedCrop?.height && imgRef.current && selectedFile) {
      generateCroppedPreview(imgRef.current, completedCrop);
    }
  }, [completedCrop, selectedFile]);

  async function generateCroppedPreview(image, cropData) { // UNCHANGED
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

  // --- REPLACED FUNCTION ---
  // This new version handles the logic for both free and premium users, and both premium options.
  const handleProcessCroppedImage = async (processType) => { 
    if (!croppedImageBlob) { setError('Please make a crop selection and ensure a preview is visible.'); return; }
    if (!appToken) { setError('Please sign in.'); return; }
    
    setIsLoading(true); 
    if (processType === 'ai') {
        setIsProcessingAI(true);
    }
    setError('');
    
    const isPremium = user?.tier === 'premium';
    let endpoint = '';
    let filenamePrefix = 'resized';

    if (isPremium) {
        if (processType === 'ai') {
            endpoint = `${API_BASE_URL}/process-image-premium/`;
            filenamePrefix = 'premium-enhanced';
        } else { // 'resize'
            endpoint = `${API_BASE_URL}/process-image-premium-resize/`;
        }
    } else {
        endpoint = `${API_BASE_URL}/process-image/`;
    }
    
    const formData = new FormData();
    const fName = croppedImageBlob.name || 'cropped_for_tv.png';
    formData.append('file', croppedImageBlob, fName);
    
    const controller = new AbortController();
    const tId = setTimeout(() => controller.abort(), 600000);

    try {
        const response = await fetch(endpoint, {method: 'POST', headers: {'Authorization': `Bearer ${appToken}`}, body: formData, signal: controller.signal});
        clearTimeout(tId);
        if (!response.ok) { 
            let errDetail = `Error ${response.status}: ${response.statusText}`; 
            if (response.status === 401) { errDetail = "Unauthorized. Session may have expired."; handleLogout(); } 
            else { try {const ed = await response.json(); errDetail = ed.detail || errDetail;} catch (e) {} }
            throw new Error(errDetail);
        }
        const imgBlob = await response.blob();
        processedImageRef.current = imgBlob; 
        setProcessedImageUrl(URL.createObjectURL(imgBlob));
        
        const bn = fName.replace(/\.[^/.]+$/, "")||'art'; 
        const fileExtension = response.headers.get('content-type')?.includes('jpeg') ? 'jpg' : 'png';
        setDownloadFileName(`${filenamePrefix}_${bn}.${fileExtension}`);

        setCurrentPage('result');
    } catch (err) { 
        clearTimeout(tId); 
        if(err.name==='AbortError'){setError('Image enhancement timed out. Try a smaller selection or image.');}
        else{setError(err.message||'Image processing failed.');} 
        setProcessedImageUrl(null);
    }
    finally { 
        setIsLoading(false);
        setIsProcessingAI(false);
    }
  };

  const handleDownload = () => { // UNCHANGED
    if (processedImageRef.current) {
      const url = URL.createObjectURL(processedImageRef.current); 
      const a = document.createElement('a'); a.href = url; a.download = downloadFileName;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url); 
    }
  };

  const renderPageContent = () => { // MODIFIED
    if (isLoading) { 
      return (
        <div className="loading-indicator">
          <Oval height={50} width={50} color="#673ab7" secondaryColor="#d1c4e9" strokeWidth={4} strokeWidthSecondary={4} ariaLabel="oval-loading" wrapperStyle={{ margin: "0 auto" }} visible={true}/>
          <p>{isProcessingAI ? "Enhancing with AI, this may take a moment..." : "Processing your image..."}</p>
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
              {/* --- THIS IS THE NEW BUTTON LOGIC --- */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
                {user?.tier === 'premium' ? (
                    <>
                      <button onClick={() => handleProcessCroppedImage('ai')} disabled={!croppedImageBlob || isLoading} className="action-button">
                        {isLoading && isProcessingAI ? 'AI Enhancing...' : 'Enhance with AI (Slower)'}
                      </button>
                      <button onClick={() => handleProcessCroppedImage('resize')} disabled={!croppedImageBlob || isLoading} className="action-button" style={{backgroundColor: '#7E57C2'}}>
                        {isLoading && !isProcessingAI ? 'Resizing...' : 'Resize to 4K (Faster)'}
                      </button>
                    </>
                ) : (
                    <button onClick={() => handleProcessCroppedImage('resize')} disabled={!croppedImageBlob || isLoading} className="action-button">
                      Resize to 4K (Free)
                    </button>
                )}
                <button onClick={() => resetAllImageStates(true)} className="secondary-action-button" disabled={isLoading}>
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
          {appToken && user && !isLoading && ( 
            <div className="user-info">
              {user.tier === 'premium' && <span title="Premium Tier" style={{color: '#ffd700', fontWeight: 'bold', fontSize: '1.2rem'}}>👑</span>}
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
            <p>
                <strong>Free Tier:</strong> Images are resized to 4K (3840x2160) using standard algorithms.
                <br/>
                <strong>Premium Tier:</strong> Images can be enhanced with AI for superior detail or quickly resized to 4K.
            </p>
        </div>
      </div>
    </div>
  );
}

export default App;