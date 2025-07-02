# Stage 1: Build the React app
FROM node:18-alpine AS build
WORKDIR /app

COPY . .

# Build from the frontend source directory
WORKDIR /app/app/frontend
RUN npm install
RUN npm run build

# Move the build folder to a known root-level location
RUN mv build /app/build

# Debug
RUN echo "✅ Build moved to /app/build" && ls -l /app/build

# Stage 2: Serve with Nginx
FROM nginx:1.25-alpine
WORKDIR /usr/share/nginx/html

# Copy from known location
COPY --from=build /app/build /usr/share/nginx/html

# Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
