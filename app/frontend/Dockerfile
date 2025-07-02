FROM node:18-alpine AS build

WORKDIR /app
COPY . .

RUN npm install
RUN npm run build

FROM nginx:1.25-alpine

WORKDIR /usr/share/nginx/html
COPY --from=build /app/build /usr/share/nginx/html

COPY nginx.conf /etc/nginx/conf.d/default.conf

# 🔧 Needed for envsubst
RUN apk add --no-cache gettext

# 🔁 Replace variables and run Nginx
CMD /bin/sh -c "envsubst '\$REACT_APP_BACKEND_HOST \$REACT_APP_BACKEND_PORT' < /etc/nginx/conf.d/default.conf > /etc/nginx/conf.d/default.conf && exec nginx -g 'daemon off;'"
