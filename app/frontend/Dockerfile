FROM node:18-alpine AS build

WORKDIR /app
COPY . .

WORKDIR /app/app/frontend
RUN npm install
RUN npm run build
RUN echo "🔍 Listing all build folders..." && find /app -type d -name build && ls -lR /app


# Debug to find actual build folder
# RUN echo "🔍 Build Output:" && find /app -type d -name build && ls -lR /app


FROM nginx:1.25-alpine
WORKDIR /usr/share/nginx/html

# 🛠️ Replace <REAL_PATH> after checking logs from above
COPY --from=build /app/app/frontend/build /usr/share/nginx/html


COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
