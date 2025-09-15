import type { NextConfig } from 'next';
const repo = 'expressivity-atlas';
const isProd = process.env.NODE_ENV === 'production';
const nextConfig: NextConfig = {
  output: 'export',
  basePath: isProd ? `/${repo}` : undefined,
  assetPrefix: isProd ? `/${repo}/` : undefined,
  images: { unoptimized: true },
  trailingSlash: true,
};
export default nextConfig;
