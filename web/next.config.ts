import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Explicitly anchor output file tracing root to this package's directory
  // to prevent Next.js from inferring the wrong workspace root when multiple
  // lockfiles exist in parent directories.
  outputFileTracingRoot: path.join(__dirname, "../"),
};

export default nextConfig;
