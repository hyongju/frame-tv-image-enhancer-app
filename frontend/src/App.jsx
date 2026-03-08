import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleLogin, googleLogout } from '@react-oauth/google';
import jwt_decode from 'jwt-decode';
import { Oval } from 'react-loader-spinner';
import ReactCrop, { centerCrop, makeAspectCrop } from 'react-image-crop';
import heic2any from 'heic2any';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8181';

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

// ============================================================
// Sub-Components
// ============================================================

function LoadingIndicator({ isProcessingAI }) {
  return (
    <div className="loading-indicator">
      <Oval
        height={50}
        width={50}
        color="#a855f7"
        secondaryColor="#4c1d95"
        strokeWidth={4}
        strokeWidthSecondary={4}
        ariaLabel="oval-loading"
        wrapperStyle={{ margin: '0 auto' }}
        visible={true}
      />
      <p>{isProcessingAI ? 'Enhancing with AI, this may take a moment...' : 'Processing your image...'}</p>
    </div>
  );
}

function AuthSection({ error, onLoginSuccess, onLoginError }) {
  return (
    <>
      <p className="auth-intro-text">
        Sign in to upscale your favorite images for your Samsung The Frame TV.
      </p>
      {error && <p className="error" style={{ marginBottom: '15px' }}>Error: {error}</p>}
      <GoogleLogin
        onSuccess={onLoginSuccess}
        onError={onLoginError}
        theme="filled_black"
        size="large"
        shape="rectangular"
      />
    </>
  );
}

function ErrorView({ error, onReset }) {
  return (
    <div>
      <p className="error">Error: {error}</p>
      <button
        onClick={onReset}
        className="action-button"
        style={{ marginTop: '10px' }}
      >
        Start Over
      </button>
    </div>
  );
}

function UploadStep({ fileInputRef, onFileChange }) {
  return (
    <div className="upload-section">
      <h2>Step 1: Upload Your Image</h2>
      <p>Choose any image you'd like to prepare for your Samsung Frame TV's Art Mode.</p>
      <input
        type="file"
        accept="image/*,.heic,.heif,image/heic,image/heif"
        onChange={onFileChange}
        ref={fileInputRef}
      />
      <div className="supported-formats">
        <span className="supported-formats-label">Supported formats:</span>
        {['JPEG', 'PNG', 'WEBP', 'HEIC', 'HEIF', 'GIF', 'BMP', 'TIFF'].map(fmt => (
          <span key={fmt} className="format-tag">{fmt}</span>
        ))}
      </div>
    </div>
  );
}

function CropStep({
  originalImagePreview, crop, setCrop, setCompletedCrop,
  imgRef, onImageLoad, aspect, croppedImagePreviewUrl,
  croppedImageBlob, isLoading, isProcessingAI, onProcess,
  onReset, user,
}) {
  if (!originalImagePreview) {
    return (
      <p>
        Please{' '}
        <button onClick={onReset} className="link-button">upload an image</button>
        {' '}first.
      </p>
    );
  }
  return (
    <div className="crop-area-container">
      <h2>Step 2: Crop for 16:9 Aspect</h2>
      <p>Drag to select the perfect 16:9 portion of your image.</p>
      <ReactCrop
        crop={crop}
        onChange={(_, pc) => setCrop(pc)}
        onComplete={(c) => setCompletedCrop(c)}
        aspect={aspect}
        minWidth={50}
        minHeight={Math.round(50 / aspect)}
      >
        <img
          ref={imgRef}
          alt="Crop area"
          src={originalImagePreview}
          onLoad={onImageLoad}
          style={{ maxHeight: '350px', maxWidth: '100%', display: 'block', margin: '0 auto', objectFit: 'contain' }}
        />
      </ReactCrop>
      <div className="crop-actions-container">
        {croppedImagePreviewUrl && (
          <div className="crop-output-preview-container">
            <h4>Cropped Preview:</h4>
            <img alt="Cropped Preview" src={croppedImagePreviewUrl} className="crop-output-image" />
          </div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
          {user?.tier === 'premium' ? (
            <>
              <button
                onClick={() => onProcess('ai')}
                disabled={!croppedImageBlob || isLoading}
                className="action-button"
              >
                {isLoading && isProcessingAI ? 'AI Enhancing...' : 'Enhance with AI (Slower)'}
              </button>
              <button
                onClick={() => onProcess('resize')}
                disabled={!croppedImageBlob || isLoading}
                className="action-button"
                style={{ background: 'linear-gradient(135deg, #4c1d95, #6d28d9)' }}
              >
                {isLoading && !isProcessingAI ? 'Resizing...' : 'Resize to 4K (Faster)'}
              </button>
            </>
          ) : (
            <button
              onClick={() => onProcess('resize')}
              disabled={!croppedImageBlob || isLoading}
              className="action-button"
            >
              Resize to 4K (Free)
            </button>
          )}
          <button
            onClick={onReset}
            className="secondary-action-button"
            disabled={isLoading}
          >
            Choose Different Image
          </button>
        </div>
      </div>
    </div>
  );
}

function ResultStep({ processedImageUrl, downloadFileName, onDownload, onReset }) {
  if (!processedImageUrl) {
    return (
      <p>
        Processing may have failed or no image is ready. Please{' '}
        <button onClick={onReset} className="link-button">start over</button>.
      </p>
    );
  }
  return (
    <>
      <div className="image-preview-container" style={{ marginTop: '10px' }}>
        <div className="image-box">
          <h2>Step 3: Your Frame TV Art!</h2>
          <img src={processedImageUrl} alt="Upscaled for Samsung Frame TV" />
        </div>
      </div>
      <div className="result-section">
        <p>
          Save your upscaled image.<br />
          Filename: <strong>{downloadFileName}</strong>
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
          <button onClick={onDownload} className="download-button">
            Download Enhanced Image
          </button>
          <button onClick={onReset} className="secondary-action-button">
            Enhance Another Image
          </button>
        </div>
      </div>
    </>
  );
}

function AppHeader({ appToken, user, isLoading, onLogout }) {
  return (
    <header className="app-header">
      <h1>Frame TV Image Enhancer</h1>
      {appToken && user && !isLoading && (
        <div className="user-info">
          {user.tier === 'premium' && (
            <span title="Premium Tier" style={{ color: '#ffd700', fontWeight: 'bold', fontSize: '1.2rem' }}>👑</span>
          )}
          {user.picture && (
            <img src={user.picture} alt={user.name || 'User'} className="user-avatar" />
          )}
          <span>Hi, {user.name || user.email}!</span>
          <button onClick={onLogout} className="logout-button">Logout</button>
        </div>
      )}
    </header>
  );
}

// ============================================================
// Main App Component
// ============================================================

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

  const handleFileChange = async (event) => {
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
      const isHeic = file.type === 'image/heic' || file.type === 'image/heif' ||
        /\.(heic|heif)$/i.test(file.name);
      if (isHeic) {
        setIsLoading(true);
        try {
          const convertedBlob = await heic2any({ blob: file, toType: 'image/jpeg', quality: 0.92 });
          const jpegBlob = Array.isArray(convertedBlob) ? convertedBlob[0] : convertedBlob;
          const jpegFile = new File([jpegBlob], file.name.replace(/\.(heic|heif)$/i, '.jpg'), { type: 'image/jpeg' });
          setSelectedFile(jpegFile);
          setOriginalImagePreview(URL.createObjectURL(jpegFile));
          setCurrentPage('crop');
        } catch (e) {
          console.error('HEIC conversion failed:', e);
          setError('Failed to convert HEIC image. Please try a JPEG or PNG file.');
          setSelectedFile(null);
          setCurrentPage('upload');
        } finally {
          setIsLoading(false);
        }
      } else {
        setSelectedFile(file); 
        setOriginalImagePreview(URL.createObjectURL(file));
        setCurrentPage('crop');
      }
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

  const renderPageContent = () => {
    if (isLoading) {
      return <LoadingIndicator isProcessingAI={isProcessingAI} />;
    }

    if (!appToken) {
      return (
        <AuthSection
          error={error}
          onLoginSuccess={handleGoogleLoginSuccess}
          onLoginError={handleGoogleLoginError}
        />
      );
    }

    if (error) {
      return <ErrorView error={error} onReset={() => resetAllImageStates(true)} />;
    }

    switch (currentPage) {
      case 'upload':
        return <UploadStep fileInputRef={fileInputRef} onFileChange={handleFileChange} />;
      case 'crop':
        return (
          <CropStep
            originalImagePreview={originalImagePreview}
            crop={crop}
            setCrop={setCrop}
            setCompletedCrop={setCompletedCrop}
            imgRef={imgRef}
            onImageLoad={onImageLoad}
            aspect={aspect}
            croppedImagePreviewUrl={croppedImagePreviewUrl}
            croppedImageBlob={croppedImageBlob}
            isLoading={isLoading}
            isProcessingAI={isProcessingAI}
            onProcess={handleProcessCroppedImage}
            onReset={() => resetAllImageStates(true)}
            user={user}
          />
        );
      case 'result':
        return (
          <ResultStep
            processedImageUrl={processedImageUrl}
            downloadFileName={downloadFileName}
            onDownload={handleDownload}
            onReset={() => resetAllImageStates(true)}
          />
        );
      default:
        return <ErrorView error="Something went wrong." onReset={() => resetAllImageStates(true)} />;
    }
  };

  return (
    <div className="App">
      <div className="container">
        <AppHeader
          appToken={appToken}
          user={user}
          isLoading={isLoading}
          onLogout={handleLogout}
        />

        <div className={!appToken && !isLoading ? 'auth-section' : 'main-content-area'}>
          {renderPageContent()}
        </div>

        <div className="disclaimer">
          <p>
            For the <strong>Free Tier</strong>, your images are quickly resized to a 4K resolution (3840x2160) perfect for your TV. The <strong>Premium Tier</strong> adds an extra step: your image is first processed by the incredible{' '}
            <a href="https://github.com/xinntao/Real-ESRGAN" target="_blank" rel="noopener noreferrer">Real-ESRGAN</a>{' '}
            AI upscaler to create stunning detail before being resized.
          </p>
          <p>
            Full credit for the AI technology goes to the original developers. Please note that the Premium Tier is currently for personal and testing purposes only due to the significant GPU resources required.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;