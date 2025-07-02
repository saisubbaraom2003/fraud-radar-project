# Stage 1: Build the frontend application
FROM node:18-alpine AS build

# Set the working directory inside the container
# This WORKDIR is crucial as it defines the base for subsequent COPY and RUN commands
WORKDIR /app/app/frontend

# Copy package.json and package-lock.json first to leverage Docker cache
COPY ./package*.json ./

# Install frontend dependencies
RUN npm install

# Copy the rest of the frontend application source code
# This copies all files from the current build context (your repo root)
# into /app/app/frontend inside this build stage.
COPY . .

# Build the frontend application
RUN npm run build


# Stage 2: Serve the frontend with Nginx
FROM nginx:1.25-alpine AS stage-1

# Install gettext for envsubst if you're using environment variables in nginx.conf
RUN apk add --no-cache gettext

# Copy the built frontend assets from the 'build' stage
# The 'build' stage outputs the compiled React/frontend app into /app/app/frontend/build
COPY --from=build /app/app/frontend/build /usr/share/nginx/html

# *** THIS IS THE CRUCIAL FIX ***
# Copy the nginx.conf template directly from your local repository's frontend directory
# Since your Dockerfile is at the repo root and nginx.conf is in 'frontend/',
# 'frontend/nginx.conf' is the correct relative path from the build context.
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf.template

# Use envsubst to replace environment variables in the nginx.conf template
# This is usually done to inject backend API URLs etc.
CMD ["/bin/sh", "-c", "envsubst < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]

# Expose port 80 (standard HTTP port for Nginx)
EXPOSE 80