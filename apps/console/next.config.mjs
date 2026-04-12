import CopyPlugin from 'copy-webpack-plugin';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cesiumSource = path.join(__dirname, 'node_modules', 'cesium', 'Build', 'Cesium');

const nextConfig = {
  output: 'standalone',
  devIndicators: false,
  outputFileTracingRoot: path.join(__dirname, '../../'),
  env: {
    CESIUM_BASE_URL: '/cesium',
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
