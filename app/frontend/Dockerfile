FROM node:18-alpine AS build

WORKDIR /app
COPY . .

# Ensure we’re in the right subfolder
WORKDIR /app/app/frontend
RUN npm install

# Run the build
RUN npm run build

# 🧪 NEW: Print the actual location of 'build' folder
RUN echo "🔍 pwd: $(pwd)" && find /app -type d -name build && ls -lR /app
