# app/frontend/Dockerfile
# Multi-stage Dockerfile for building and serving the React application

### STAGE 1: Build the React application ###
FROM node:18-alpine AS build
# Using Node.js 18 Alpine as the base image for building

WORKDIR /app
# Set the working directory inside the build container

# Copy ALL application source code first
# This ensures all files, including package.json, are present before npm install
COPY . . # <--- MOVED THIS LINE UP

# --- AGGRESSIVE NPM INSTALL FIX / DIAGNOSTICS ---
# List contents of /app to confirm package.json is there BEFORE npm install
RUN ls -l /app

# Clean npm cache forcefully to prevent any corrupted cache issues
RUN npm cache clean --force

# Remove any existing node_modules directory to ensure a fresh install
RUN rm -rf node_modules

# Install frontend dependencies
RUN npm install

# Build the React application for production
# This command typically creates a 'build' directory with static assets
RUN npm run build

### STAGE 2: Serve the application with Nginx ###
FROM nginx:1.25-alpine
# Using Nginx Alpine as the base image for serving the static files

# Set the working directory to the Nginx HTML root for clarity
WORKDIR /usr/share/nginx/html

# Copy the production build output from the 'build' stage
# The /app/build directory from the first stage is copied to Nginx's html directory
COPY --from=build /app/build /usr/share/nginx/html

# Copy the custom Nginx configuration using its full path relative to the repo root
COPY app/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80, which Nginx listens on by default
EXPOSE 80

# The command to start Nginx when the container launches
# 'daemon off;' keeps Nginx running in the foreground, essential for Docker containers
CMD ["nginx", "-g", "daemon off;"]
