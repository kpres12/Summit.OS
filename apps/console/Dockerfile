# Use a lightweight Node image suitable for Next.js dev
FROM node:18-alpine

WORKDIR /app

# Install dependencies based on lockfiles when available
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN if [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then npm install -g pnpm && pnpm i --frozen-lockfile; \
    elif [ -f yarn.lock ]; then npm install -g yarn && yarn install --frozen-lockfile; \
    else npm install; fi

# Copy the rest of the app (source will be bind-mounted in dev, this is for image completeness)
COPY . .

EXPOSE 3000

# Ensure node_modules exists even when it's a named volume in compose
CMD ["sh", "-c", "[ -d node_modules ] || npm install && npm run dev"]
