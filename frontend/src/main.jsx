import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';
import { GoogleOAuthProvider } from '@react-oauth/google';

const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;

if (!googleClientId) {
  console.error("ERROR: VITE_GOOGLE_CLIENT_ID is not defined. Please set it in your .env file.");
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <GoogleOAuthProvider clientId={googleClientId || "YOUR_FALLBACK_CLIENT_ID_IF_ENV_FAILS"}>
      <App />
    </GoogleOAuthProvider>
  </React.StrictMode>,
);