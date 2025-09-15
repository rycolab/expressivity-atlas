// next.config.ts
import type { NextConfig } from 'next'

const repo = 'expressivity-atlas'
const isProd = process.env.NODE_ENV === 'production'

const nextConfig: NextConfig = {
  output: 'export',                 // static export
  basePath: isProd ? `/${repo}` : undefined,   // needed for GH Pages subpath
  assetPrefix: isProd ? `/${repo}/` : undefined,
  images: { unoptimized: true },    // Next/Image needs this when exporting
}
export default nextConfig
