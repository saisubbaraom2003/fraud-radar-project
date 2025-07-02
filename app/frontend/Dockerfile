# app/frontend/Dockerfile

### Stage 1: Build React App ###
FROM node:18-alpine AS build

WORKDIR /app
COPY . .

# Install dependencies and build app
RUN npm install && npm run build


### Stage 2: Serve With Nginx ###
FROM nginx:1.25-alpine

# Install envsubst for variable substitution
RUN apk add --no-cache gettext

# Copy React production build
COPY --from=build /app/build /usr/share/nginx/html

# Copy Nginx config template
COPY nginx.conf /etc/nginx/conf.d/default.conf.template

# Replace env vars and run Nginx
CMD ["/bin/sh", "-c", "envsubst '$REACT_APP_BACKEND_HOST $REACT_APP_BACKEND_PORT' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"]
