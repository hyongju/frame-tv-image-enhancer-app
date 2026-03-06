# Deployment Instructions for Image Enhancer Service

This directory contains the systemd service unit for the image processing application (named `imageprocessing.service`) and instructions on how to deploy and manage it.

## Files

- `imageprocessing.service` – the systemd unit file that uses Docker Compose to manage the application.

## Deployment Steps

1. **Copy the service file**

   Place `imageenhancer.service` into the systemd system directory (usually `/etc/systemd/system/`) on the target machine:

   ```bash
   sudo cp ./deploy/imageenhancer.service /etc/systemd/system/
   sudo systemctl daemon-reload
   ```

2. **Enable the service**

   Enable the service to start automatically at boot:

   ```bash
   sudo systemctl enable imageenhancer.service
   ```

3. **Start the service**

   ```bash
   sudo systemctl start imageenhancer.service
   ```

4. **Check status and logs**

   ```bash
   sudo systemctl status imageenhancer.service
   sudo journalctl -u imageenhancer.service -f
   ```

## Maintenance Commands

- **Stop the service**
  ```bash
  sudo systemctl stop imageenhancer.service
  ```

- **Restart the service**
  ```bash
  sudo systemctl restart imageenhancer.service
  ```

- **Reload configuration** (after editing the unit file):
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl restart imageenhancer.service
  ```

- **Disable autostart**
  ```bash
  sudo systemctl disable imageenhancer.service
  ```

## Notes

- The `ExecStart` path assumes a Python virtual environment located in the project root under `.venv`. Adjust the `User`, `WorkingDirectory`, and command-line options as necessary for your deployment environment.
- For development you may want to run the app manually with uvicorn; systemd is intended for production.
- Always run `sudo systemctl daemon-reload` after modifying the unit file.
