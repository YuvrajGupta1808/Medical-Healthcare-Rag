# Stage 1: Build dependencies
FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies first for caching layers
COPY package*.json ./
RUN npm install --legacy-peer-deps

# Copy the rest and build
COPY tsconfig.json next.config.mjs postcss.config.js tailwind.config.js ./
COPY src ./src
COPY public ./public

RUN npm run build

# Stage 2: Runner
FROM node:18-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

# Copy built assets
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["npx", "next", "start"]
