# Stage 1: Build the React application
FROM node:18-alpine AS build
WORKDIR /app

# Copy all repo files to /app
COPY . .

# Move into the actual frontend folder
WORKDIR /app/app/frontend

# Install dependencies
RUN npm install

# Debug before build
RUN echo "🔧 Current directory: $(pwd)" && ls -l

# Build app and check for errors
RUN npm run build || (echo "❌ Build failed" && exit 1)

# Debug after build
RUN echo "✅ Directory after build:" && ls -l

# Stage 2: Serve the application with Nginx
FROM nginx:1.25-alpine
WORKDIR /usr/share/nginx/html

# Copy build output
COPY --from=build /app/app/frontend/build /usr/share/nginx/html

# Copy and apply custom Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
