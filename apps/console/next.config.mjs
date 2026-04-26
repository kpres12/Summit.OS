import CopyPlugin from 'copy-webpack-plugin';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cesiumSource = path.join(__dirname, 'node_modules', 'cesium', 'Build', 'Cesium');

// Security headers applied to every response. Tightened for production
// since Heli.OS is a closed proprietary SaaS — no third-party embeds
// expected outside of CesiumJS workers (which are same-origin).
const securityHeaders = [
  {
    key: 'Content-Security-Policy',
    value: [
      "default-src 'self'",
      // Cesium needs blob: workers + WebAssembly + inline styles for its
      // canvas widgets. Tighten further if/when Cesium is replaced.
      "script-src 'self' 'wasm-unsafe-eval' blob:",
      "worker-src 'self' blob:",
      "style-src 'self' 'unsafe-inline' fonts.googleapis.com",
      "font-src 'self' fonts.gstatic.com data:",
      "img-src 'self' data: blob: https:",
      "connect-src 'self' wss: https:",
      "object-src 'none'",
      "base-uri 'self'",
      "form-action 'self'",
      "frame-ancestors 'none'",
      "upgrade-insecure-requests",
    ].join('; '),
  },
  {
    key: 'Strict-Transport-Security',
    value: 'max-age=63072000; includeSubDomains; preload',
  },
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  {
    key: 'Permissions-Policy',
    value: 'camera=(), microphone=(), geolocation=(self), payment=(), usb=()',
  },
  { key: 'X-DNS-Prefetch-Control', value: 'off' },
];

const nextConfig = {
  output: 'standalone',
  devIndicators: false,
  images: { unoptimized: true },
  poweredByHeader: false,
  env: {
    CESIUM_BASE_URL: '/cesium',
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: securityHeaders,
      },
    ];
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.plugins.push(
        new CopyPlugin({
          patterns: [
            { from: path.join(cesiumSource, 'Workers'), to: path.join(__dirname, 'public', 'cesium', 'Workers') },
            { from: path.join(cesiumSource, 'ThirdParty'), to: path.join(__dirname, 'public', 'cesium', 'ThirdParty') },
            { from: path.join(cesiumSource, 'Assets'), to: path.join(__dirname, 'public', 'cesium', 'Assets') },
            { from: path.join(cesiumSource, 'Widgets'), to: path.join(__dirname, 'public', 'cesium', 'Widgets') },
          ],
        })
      );
    }
    return config;
  },
};

export default nextConfig;
