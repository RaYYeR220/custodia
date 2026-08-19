import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // pin the trace root to this app: without it Next walks up past the repo and
  // picks up an unrelated lockfile, which changes what lands in the standalone
  // bundle the container runs
  outputFileTracingRoot: path.join(__dirname),
  reactStrictMode: true,
};

export default nextConfig;
