### STAGE 1: Build the React Application ###
FROM node:18-alpine AS build

WORKDIR /app
COPY . .

WORKDIR /app/app/frontend
RUN npm install
RUN npm run build


### STAGE 2: Serve with Nginx ###
FROM nginx:1.25-alpine

WORKDIR /usr/share/nginx/html

# ✅ Correct path to React build
COPY --from=build /app/app/frontend/build /usr/share/nginx/html

# ✅ Fix: use correct relative path for nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
