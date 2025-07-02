# Stage 1: Build React app
FROM node:18-alpine AS build

WORKDIR /app
COPY . .

RUN npm install
RUN npm run build

# Stage 2: Serve using Nginx
FROM nginx:1.25-alpine

WORKDIR /usr/share/nginx/html

# ✅ Copy the correct path for build folder
COPY --from=build /app/build /usr/share/nginx/html

# ✅ Use the correct nginx.conf path (relative to Docker context)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
