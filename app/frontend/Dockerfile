# app/frontend/Dockerfile
# Multi-stage Dockerfile for building and serving the React application

### STAGE 1: Build the React application ###
FROM node:18-alpine AS build
# Using Node.js 18 Alpine as the base image for building

WORKDIR /app
# Set the initial working directory inside the build container to copy the whole repo

# Copy ALL application source code first from the build context (which is the repo root)
# This ensures all files, including the nested frontend directory, are present.
COPY . .

# Change the working directory to the actual location of the frontend application
# Now, /app/app/frontend is where package.json and source code reside within the container.
WORKDIR /app/app/frontend # <--- CRITICAL FIX: Change WORKDIR to the actual frontend path

# Install frontend dependencies
# npm install will now find package.json in the current working directory (/app/app/frontend)
RUN npm install

# Build the React application for production
RUN npm run build

### STAGE 2: Serve the application with Nginx ###
FROM nginx:1.25-alpine
# Using Nginx Alpine as the base image for serving the static files

# Set the working directory to the Nginx HTML root
WORKDIR /usr/share/nginx/html

# Copy the production build output from the 'build' stage
# The /app/app/frontend/build directory from the first stage is copied to Nginx's html directory
COPY --from=build /app/app/frontend/build /usr/share/nginx/html # <--- Updated source path for build output

# Copy the custom Nginx configuration using its full path relative to the repo root
COPY app/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80, which Nginx listens on by default
EXPOSE 80

# The command to start Nginx when the container launches
# 'daemon off;' keeps Nginx running in the foreground, essential for Docker containers
CMD ["nginx", "-g", "daemon off;"]
