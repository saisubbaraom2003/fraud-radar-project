# Stage 1: Build React App
FROM node:18-alpine as build

WORKDIR /app
COPY . .
RUN npm install
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:1.25-alpine
RUN apk add --no-cache gettext

# Copy build files from Stage 1
COPY --from=build /app/build /usr/share/nginx/html

# Copy Nginx config template and use envsubst
COPY nginx.conf /etc/nginx/conf.d/default.conf.template

CMD ["/bin/sh", "-c", "envsubst '$REACT_APP_BACKEND_HOST $REACT_APP_BACKEND_PORT' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"]
