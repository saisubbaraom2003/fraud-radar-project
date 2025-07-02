# app/frontend/Dockerfile
# Multi-stage Dockerfile for building and serving the React application

### STAGE 1: Build the React application ###
FROM node:18-alpine AS build
# Using Node.js 18 Alpine as the base image for building

WORKDIR /app
# Set the working directory inside the container to /app.
# Since the build context is ./app/frontend (from render.yaml),
# COPY . . will copy the contents of app/frontend directly into /app.

# Copy ALL application source code from the build context (which is ./app/frontend)
# into the /app directory inside the container.
COPY . .

# Change the working directory to the actual location of the frontend application
# Now, /app/app/frontend is where package.json and source code reside within the container.
WORKDIR /app/app/frontend

# Install frontend dependencies
RUN npm install

# Build the React application for production
# This command typically creates a 'build' directory within /app/app/frontend
RUN npm run build

### STAGE 2: Serve the application with Nginx ###
FROM nginx:1.25-alpine
# Using Nginx Alpine as the base image for serving the static files

# Explicitly create the Nginx HTML directory to ensure it exists
RUN mkdir -p /usr/share/nginx/html

# Set the working directory to the Nginx HTML root for clarity
WORKDIR /usr/share/nginx/html

# Copy the production build output from the 'build' stage
# The /app/build directory from the first stage is copied to Nginx's html directory
COPY --from=build /app/build /usr/share/nginx/html

# --- CRITICAL FIX: Directly write Nginx config and override CMD ---
# Remove the default Nginx configuration to prevent conflicts
RUN rm -f /etc/nginx/conf.d/default.conf

# Write our custom Nginx configuration directly into the file
RUN echo 'server { \
  listen 80; \
  root /usr/share/nginx/html; \
  index index.html; \
  \
  location / { \
    try_files $uri /index.html; \
  } \
  \
  location /api { \
    proxy_pass http://fraudradar-backend:8000; \
    proxy_set_header Host $host; \
    proxy_set_header X-Real-IP $remote_addr; \
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
    proxy_set_header X-Forwarded-Proto $scheme; \
  } \
}' > /etc/nginx/conf.d/default.conf

# Expose port 80, which Nginx listens on by default
EXPOSE 80

# Override the default Nginx CMD to explicitly use our configuration
# This bypasses Render's docker-entrypoint.sh if it's causing issues.
CMD ["nginx", "-g", "daemon off;"] # <--- Overriding CMD

