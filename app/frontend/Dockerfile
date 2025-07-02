### STAGE 1: Build the React Application ###
FROM node:18-alpine AS build

# Set working directory for build stage
WORKDIR /app

# Copy the entire repo into the container
COPY . .

# Navigate to frontend source directory
WORKDIR /app/app/frontend

# Install frontend dependencies
RUN npm install

# Build the React app for production
RUN npm run build

# (Optional) Debug: Uncomment if needed
# RUN find /app -type d -name build && ls -lR /app/app/frontend/build


### STAGE 2: Serve the Build with Nginx ###
FROM nginx:1.25-alpine

# Set working directory for Nginx
WORKDIR /usr/share/nginx/html

# Copy the React build output from the previous stage
COPY --from=build /app/app/frontend/build /usr/share/nginx/html

# Copy custom Nginx config if you have one
COPY app/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose default Nginx port
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
