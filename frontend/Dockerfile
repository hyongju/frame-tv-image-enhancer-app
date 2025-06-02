# Stage 1: Build the React app with Vite
FROM node:18-alpine as builder

WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock, pnpm-lock.yaml)
COPY package*.json ./
# If using pnpm, copy pnpm-lock.yaml as well
# COPY pnpm-lock.yaml ./

# Install dependencies
# If using npm:
RUN npm install
# If using yarn:
# RUN yarn install
# If using pnpm:
# RUN npm install -g pnpm && pnpm install --frozen-lockfile

# Copy the rest of the application code
COPY . .

# Set the API URL build argument. Vite uses VITE_ prefix.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

# Build the Vite app for production (output in /app/dist)
RUN npm run build
# or yarn build or pnpm build

# Stage 2: Serve the static files with Nginx
FROM nginx:1.25-alpine

# Remove default Nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy static assets from builder stage (Vite builds to 'dist' folder)
COPY --from=builder /app/dist /usr/share/nginx/html

# Optional: Copy custom Nginx configuration for SPA routing
# If you have frontend/nginx.conf (same as provided before), uncomment this.
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 (Nginx default)
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]