import { fileURLToPath } from 'node:url';

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  outputFileTracingRoot: fileURLToPath(new URL('.', import.meta.url)),
  assetPrefix: '.',
  trailingSlash: true,
  images: { unoptimized: true },
};
export default config;
