# app/frontend/Dockerfile

### STAGE 1: Build the React application ###
FROM node:18-alpine AS build

WORKDIR /app

# Copy package.json and package-lock.json for dependency installation
COPY package*.json ./
RUN npm install

# Copy the rest of the application source code
# This copies everything from 'app/frontend' into /app in the container
COPY . .

# Build the application for production
RUN npm run build

### STAGE 2: Serve the application with Nginx ###
FROM nginx:1.25-alpine

# Copy the production build from the 'build' stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy the custom Nginx configuration
# This path assumes your nginx.conf is directly in app/frontend/
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 (Nginx default)
EXPOSE 80

# The command to start Nginx when the container launches
CMD ["nginx", "-g", "daemon off;"]