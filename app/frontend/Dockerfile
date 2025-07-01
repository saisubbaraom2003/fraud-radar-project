# app/frontend/Dockerfile
# Multi-stage Dockerfile for building and serving the React application

### STAGE 1: Build the React application ###
FROM node:18-alpine AS build
# Using Node.js 18 Alpine as the base image for building

WORKDIR /app
# Set the working directory inside the build container

# Copy package.json and package-lock.json first to leverage Docker cache
# This allows npm install to be cached if dependencies don't change
COPY package*.json ./

# Install frontend dependencies
RUN npm install

# Copy the rest of the application source code
# This copies everything from the 'app/frontend' directory into /app in the container
COPY . .

# Build the React application for production
# This command typically creates a 'build' directory with static assets
RUN npm run build

### STAGE 2: Serve the application with Nginx ###
FROM nginx:1.25-alpine
# Using Nginx Alpine as the base image for serving the static files

# Set the working directory to the Nginx HTML root for clarity in diagnostic
WORKDIR /usr/share/nginx/html

# Copy the production build output from the 'build' stage
# The /app/build directory from the first stage is copied to Nginx's html directory
COPY --from=build /app/build /usr/share/nginx/html

# --- TEMPORARY DIAGNOSTIC STEP ---
# This command will list the contents of the /usr/share/nginx/html directory
# and the root of the build context (which is where nginx.conf should be relative to).
# This helps us see if nginx.conf is actually available in the context.
RUN ls -R /usr/share/nginx/html && ls -R /

# Copy the custom Nginx configuration
# This configuration file will tell Nginx how to serve the React app and proxy API requests
# This path assumes your nginx.conf is directly in app/frontend/ and will be copied to default location
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80, which Nginx listens on by default
EXPOSE 80

# The command to start Nginx when the container launches
# 'daemon off;' keeps Nginx running in the foreground, essential for Docker containers
CMD ["nginx", "-g", "daemon off;"]
