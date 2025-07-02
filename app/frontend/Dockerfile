# Stage 1: Build React App
FROM node:18-alpine AS build

# Set working directory
WORKDIR /app

# Copy entire repo (from root) so we can access nested /app/frontend
COPY . .

# Change to frontend directory
WORKDIR /app/app/frontend

# Install and build
RUN npm install && npm run build

# Stage 2: Serve with Nginx
FROM nginx:1.25-alpine

# Add envsubst for variable substitution
RUN apk add --no-cache gettext

# Copy built frontend app from previous stage
COPY --from=build /app/app/frontend/build /usr/share/nginx/html

# Copy nginx.conf template
COPY app/frontend/nginx.conf /etc/nginx/conf.d/default.conf.template

# Use envsubst to replace backend host/port from environment variables
CMD ["/bin/sh", "-c", "envsubst '\$REACT_APP_BACKEND_HOST \$REACT_APP_BACKEND_PORT' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"]
