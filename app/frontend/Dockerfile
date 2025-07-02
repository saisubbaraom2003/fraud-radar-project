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
# This addresses the "no such file or directory" error for /usr/share/nginx
RUN mkdir -p /usr/share/nginx/html # <--- ADDED THIS LINE

# Set the working directory to the Nginx HTML root for clarity
WORKDIR /usr/share/nginx/html

# Copy the production build output from the 'build' stage
# The /app/build directory from the first stage is copied to Nginx's html directory
COPY --from=build /app/build /usr/share/nginx/html

# Copy the custom Nginx configuration.
# Since dockerContext is ./app/frontend, nginx.conf is at the root of the context.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80, which Nginx listens on by default
EXPOSE 80

# The command to start Nginx when the container launches
# 'daemon off;' keeps Nginx running in the foreground, essential for Docker containers
CMD ["nginx", "-g", "daemon off;"]
