FROM node:24-bullseye-slim

WORKDIR /app

# Install basic tools needed to fetch Bun
RUN apt-get update && apt-get install -y curl ca-certificates --no-install-recommends \
	&& rm -rf /var/lib/apt/lists/*

# Copy package files and install dependencies with Bun
COPY package*.json ./

# Install Bun and add to PATH
RUN curl -fsSL https://bun.sh/install | bash -s -- --prefix /usr/local \
	&& ln -s /root/.bun/bin/bun /usr/local/bin/bun
ENV PATH="/root/.bun/bin:${PATH}"

# Use Bun for dependency install
RUN bun install

# Copy source code
COPY . .

# Build the React app via Vite using existing npm scripts via Bun
RUN bun run build

# Install a simple static file server to serve the built assets
RUN npm install -g serve

# Expose port used by the static server
EXPOSE 5173

# Serve the dist folder
CMD ["serve", "-s", "dist", "-l", "5173"]
