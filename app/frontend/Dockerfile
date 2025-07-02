# Stage 1: Build React App
FROM node:18-alpine AS build

WORKDIR /app
COPY . .

RUN npm install && npm run build

# Stage 2: Serve with Nginx
FROM nginx:1.25-alpine

WORKDIR /usr/share/nginx/html
RUN apk add --no-cache gettext

# Copy the built React files
COPY --from=build /app/build /usr/share/nginx/html

# Copy Nginx config template
COPY nginx.conf /etc/nginx/conf.d/default.conf.template

# Replace env vars in Nginx config and start Nginx
CMD ["/bin/sh", "-c", "envsubst '$REACT_APP_BACKEND_HOST $REACT_APP_BACKEND_PORT' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"]
