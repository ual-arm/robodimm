# Robodimm Frontend Production Dockerfile
# Build from project root:
#   docker build -t robodimm/frontend .
# Run container:
#   docker run -p 8080:80 robodimm/frontend

FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
# Uses npm ci for reproducible builds (requires package-lock.json in repo), or change to "npm install" if not tracking the lockfile
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
