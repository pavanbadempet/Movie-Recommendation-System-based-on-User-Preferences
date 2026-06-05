FROM node:24-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build the React app via Vite
RUN npm run build

# Install a simple static file server to serve the built assets
RUN npm install -g serve

# Expose port
EXPOSE 5173

# Serve the dist folder
CMD ["serve", "-s", "dist", "-l", "5173"]
