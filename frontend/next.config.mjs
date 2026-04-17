/** @type {import('next').NextConfig} */
const nextConfig = {
  distDir: '../.next',
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
