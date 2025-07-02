# app/frontend/Dockerfile
# Multi-stage Dockerfile for building and serving the React application

### STAGE 1: Build the React application ###
FROM node:18-alpine AS build
# Using Node.js 18 Alpine as the base image for building

WORKDIR /app
# Set the initial working directory inside the build container to copy the whole repo

# Copy ALL application source code first from the build context (which is 'app/frontend' due to render.yaml)
COPY . .

# Change the working directory to the actual location of the frontend application
# Now, /app/app/frontend is where package.json and source code reside within the container.
WORKDIR /app/app/frontend

# Install frontend dependencies
RUN npm install

# Build the React application for production
RUN npm run build

### STAGE 2: Serve the application with Nginx ###
FROM nginx:1.25-alpine
# Using Nginx Alpine as the base image for serving the static files

# Set the working directory to the Nginx HTML root for clarity
WORKDIR /usr/share/nginx/html

# Copy the production build output from the 'build' stage
# The /app/app/frontend/build directory from the first stage is copied to Nginx's html directory
COPY --from=build /app/app/frontend/build /usr/share/nginx/html

# Copy the custom Nginx configuration.
# Since dockerContext is ./app/frontend, nginx.conf is at the root of the context.
COPY nginx.conf /etc/nginx/conf.d/default.conf # <--- CORRECTED THIS LINE

# Expose port 80, which Nginx listens on by default
EXPOSE 80

# The command to start Nginx when the container launches
# 'daemon off;' keeps Nginx running in the foreground, essential for Docker containers
CMD ["nginx", "-g", "daemon off;"]
