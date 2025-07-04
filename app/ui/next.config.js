/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === 'production';
const repoName = 'expressivity-atlas';

module.exports = {
  output: 'export',           // Enable static export
  basePath: isProd ? `/${repoName}` : '',
  assetPrefix: isProd ? `/${repoName}/` : '',
  trailingSlash: true,
};
