# Frame TV Image Enhancer

Upscale your favorite images to stunning 4K quality, perfectly optimized for display on your Samsung The Frame TV's Art Mode. This application uses advanced AI (Real-ESRGAN) on the backend to enhance your images, with a user-friendly React frontend for uploading, cropping, and downloading your artwork.

**Live Demo:** https://image.hyongju.com
**API Docs:** https://api.hyongju.com/docs

![image](https://github.com/user-attachments/assets/fc3199bd-165c-4be4-b851-a8a56e5775c2)

## Features

*   **AI Image Upscaling:** Enhances image resolution using Real-ESRGAN for crisp, clear results.
*   **User Authentication:** Secure sign-in with Google.
*   **Image Cropping:** Interactive 16:9 aspect ratio cropping tool to perfectly frame your art.
*   **Image Preview:** See your original image, the cropped selection, and the final enhanced result.
*   **Download Enhanced Images:** Easily download your upscaled artwork.
*   **Responsive Design:** Works on desktop and mobile browsers.
*   **Mobile PWA Ready:** Supports "Add to Home Screen" on iOS and Android for an app-like experience.

## Tech Stack

*   **Frontend:**
    *   React (with Vite for bundling)
    *   JavaScript
    *   `react-image-crop` for image cropping
    *   `@react-oauth/google` for Google Sign-In
    *   `react-loader-spinner` for loading animations
    *   CSS3 for styling
*   **Backend:**
    *   FastAPI (Python web framework)
    *   Real-ESRGAN (via `realesrgan` and `basicsr` Python libraries) for AI upscaling
    *   Pillow & OpenCV for image manipulation
    *   `python-jose` for JWT authentication (app tokens)
    *   `google-auth` for verifying Google ID tokens
*   **Deployment (Example):**
    *   Docker & Docker Compose for containerization and orchestration
    *   Nginx (for serving frontend static files in Docker)
    *   Uvicorn (for running FastAPI backend)
    *   (Optional: Mention cloud provider, CI/CD tools if used)

## Project Structure

```
image-processor-app/
├── backend/
│   ├── app/                 # FastAPI application code (main.py)
│   ├── realesrgan/          # Local Real-ESRGAN module (if used locally)
│   ├── weights/             # Pre-trained Real-ESRGAN models (.pth files)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example         # Example environment variables for backend
├── frontend/
│   ├── public/
│   ├── src/                 # React application code (App.jsx, App.css)
│   ├── Dockerfile
│   ├── nginx.conf           # (Optional) Nginx config for serving React build
│   ├── package.json
│   └── .env.example         # Example environment variables for frontend build
├── docker-compose.yml
├── .env.example             # Example environment variables for docker-compose
└── README.md
```

## Prerequisites

*   Node.js (v18+ recommended) and npm/yarn
*   Python (v3.10+ recommended for backend)
*   Docker and Docker Compose
*   (For GPU backend) NVIDIA GPU with appropriate drivers and NVIDIA Container Toolkit installed on the host.
*   Google Cloud Platform Project with OAuth 2.0 Client ID configured.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone [Your Repository URL]
cd image-processor-app
```

### 2. Backend Configuration

*   Navigate to the `backend` directory: `cd backend`
*   **Create `.env` file:** Copy `backend/.env.example` to `backend/.env` and fill in your actual credentials:
    ```env
    # backend/.env
    GOOGLE_CLIENT_ID="YOUR_GOOGLE_CLIENT_ID_FROM_GCP"
    SECRET_KEY="your-very-strong-random-secret-key-for-jwt"
    ALGORITHM="HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES=60 
    # Add any Apple Sign-In variables here if you re-add that feature
    ```
*   **Download Real-ESRGAN Models:**
    Download the required `.pth` model files (e.g., `RealESRGAN_x4plus.pth`) from the official Real-ESRGAN repository or other sources. Place them into the `backend/weights/` directory.
    *(You might want to provide direct links or a script if these are not included in the repo).*
*   **Python Dependencies (if running locally without Docker):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

### 3. Frontend Configuration

*   Navigate to the `frontend` directory: `cd ../frontend`
*   **Install Dependencies:**
    ```bash
    npm install
    # or
    # yarn install
    ```
*   **Create `.env` file for local development:** Copy `frontend/.env.example` to `frontend/.env` (or `frontend/.env.local`) and update values:
    ```env
    # frontend/.env or .env.local
    VITE_API_URL=http://localhost:8181 # Backend URL for local frontend dev
    VITE_GOOGLE_CLIENT_ID="YOUR_GOOGLE_CLIENT_ID_FROM_GCP"
    # Add any Apple Sign-In client/service IDs here if re-added
    ```

### 4. Docker Compose Environment Configuration

*   In the project root directory (`image-processor-app/`), copy `.env.example` to `.env`.
*   This file is used by `docker-compose.yml` to substitute variables, especially for build arguments.
    ```env
    # image-processor-app/.env (root project directory)
    VITE_API_URL_FOR_COMPOSE=https://api.hyongju.com # Or http://localhost:8181 for local Docker setup
    VITE_GOOGLE_CLIENT_ID_FROM_HOST_ENV="YOUR_GOOGLE_CLIENT_ID_FROM_GCP"
    # Add Apple vars here if re-added, e.g., APPLE_SERVICES_ID_FROM_HOST_ENV=...

    # For Linux hosts to fix volume permissions if needed when running containers as non-root
    # Export these in your shell before running 'docker-compose up' or define them here if static
    # MY_UID=1000 
    # MY_GID=1000 
    ```
    *Note on `MY_UID`/`MY_GID`*: If you're using the `user: "${MY_UID}:${MY_GID}"` in `docker-compose.yml` for the backend, ensure these are correctly set in your shell environment or defined in this root `.env` file if your Docker setup requires it.

## Running the Application

### Using Docker Compose (Recommended)

This is the easiest way to run both frontend and backend together.

1.  **Ensure Docker Desktop or Docker Engine is running.**
2.  From the project root directory (`image-processor-app/`):
    ```bash
    # To run as your current host user (primarily for Linux, ensure MY_UID/MY_GID are exported)
    # export MY_UID=$(id -u)
    # export MY_GID=$(id -g)
    
    docker-compose up --build -d
    ```
    *   `--build`: Forces Docker to rebuild the images if there are changes in Dockerfiles or related source code.
    *   `-d`: Runs in detached mode (in the background).

3.  **Accessing the app:**
    *   **Frontend:** `http://localhost:3131` (or the port you configured for the frontend in `docker-compose.yml`)
    *   **Backend API Docs:** `http://localhost:8181/docs` (or the host port for the backend)

4.  **To stop the application:**
    ```bash
    docker-compose down
    ```

5.  **To view logs:**
    ```bash
    docker-compose logs -f backend
    docker-compose logs -f frontend
    ```

### Running Frontend and Backend Separately (for Development)

*   **Backend (FastAPI):**
    ```bash
    cd backend
    # source venv/bin/activate # If using a virtual environment
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 
    # Access at http://localhost:8000
    ```
    *(Ensure RealESRGAN model weights are in `backend/weights/` and all dependencies from `requirements.txt` are installed).*

*   **Frontend (React + Vite):**
    ```bash
    cd frontend
    npm run dev
    # Access at http://localhost:5173 (or whatever port Vite serves on)
    ```
    *(Ensure `VITE_API_URL` in `frontend/.env.local` points to `http://localhost:8000` if the backend is running locally on that port).*


## Making it a PWA (Add to Home Screen)

The application includes meta tags and (optionally) a `manifest.json` to support "Add to Home Screen" functionality on iOS and Android, providing a more app-like experience.
*   **iOS:** Use Safari's "Share" -> "Add to Home Screen" option. Ensure `apple-touch-icon` links in `frontend/public/index.html` point to valid icon files.
*   **Android:** Chrome should prompt to "Add to Home screen" if PWA criteria are met.

## Production Deployment (Conceptual)

*   Build production Docker images.
*   Push images to a container registry (e.g., Docker Hub, AWS ECR, GCP Artifact Registry).
*   Deploy containers to a cloud platform (e.g., AWS ECS/EKS, GCP Cloud Run/GKE, Azure AKS/Container Instances) or your own server.
*   Configure a reverse proxy (Nginx, Caddy, or cloud load balancer) for SSL termination, custom domains, and routing traffic to your frontend and backend services.
*   Set production environment variables securely (e.g., using secrets management services).
*   Ensure backend's CORS `origins` list includes your production frontend domain.
*   For GPU-accelerated backend, ensure your deployment environment supports NVIDIA GPUs.

## Auto-Start on Server Reboot (using `systemd`)

To make the Docker Compose application start automatically after a server reboot (on Linux hosts using `systemd`):

1.  Create a service file, e.g., `/etc/systemd/system/frame-tv-enhancer.service`:
    ```ini
    [Unit]
    Description=Frame TV Image Enhancer Docker Compose Service
    Requires=docker.service
    After=docker.service network-online.target

    [Service]
    Type=oneshot
    RemainAfterExit=yes
    # User=your_deploy_user # Optional: if not root
    # Group=your_deploy_group # Optional
    WorkingDirectory=/path/to/your/image-processor-app # IMPORTANT: Absolute path
    ExecStart=/usr/local/bin/docker-compose up -d --remove-orphans
    ExecStop=/usr/local/bin/docker-compose down
    TimeoutStartSec=0
    TimeoutStopSec=2min

    [Install]
    WantedBy=multi-user.target
    ```
    *(Verify path to `docker-compose` using `which docker-compose`. If using Docker Compose V2 plugin, use `docker compose` commands).*

2.  Enable and start the service:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable frame-tv-enhancer.service
    sudo systemctl start frame-tv-enhancer.service
    sudo systemctl status frame-tv-enhancer.service
    ```

## Acknowledgements

*   The AI image upscaling functionality in the backend is powered by **Real-ESRGAN**. Full credit for the core AI model and its development goes to the original authors. Please visit their repository for more information: [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
*   Samsung The Frame TV is a trademark of Samsung Electronics Co., Ltd. This application is not affiliated with Samsung.

## Contributing

(Add guidelines here if you plan to accept contributions: how to report bugs, suggest features, submit pull requests, coding standards, etc.)

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License

(Choose a license, e.g., MIT, Apache 2.0. If you don't have one, you can omit this or state "All Rights Reserved" for now.)
Distributed under the MIT License. See `LICENSE` file for more information (if you add a LICENSE file).
```

