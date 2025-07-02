# Stage 1: Build the React app
FROM node:18-alpine AS build

WORKDIR /app
COPY . .

# Enter the frontend folder and build
WORKDIR /app/app/frontend
RUN npm install
RUN npm run build

# Stage 2: Serve the app using Nginx
FROM nginx:1.25-alpine

# Install envsubst for runtime variable substitution
RUN apk add --no-cache gettext

# Copy React build from previous stage
COPY --from=build /app/app/frontend/build /usr/share/nginx/html

# Copy nginx.conf template
COPY nginx.conf /etc/nginx/conf.d/default.conf.template

# Replace env vars in nginx.conf at container startup
CMD /bin/sh -c "envsubst '\$REACT_APP_BACKEND_HOST \$REACT_APP_BACKEND_PORT' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"
