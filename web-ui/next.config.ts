import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  // Default position (bottom-left) overlaps the sidebar's user-nav area
  // (theme toggle / login-logout). top-right instead clustered it with the
  // GitHub button in chat-header.tsx, which also sits top-right — this
  // floating dev-tools badge is framework-rendered (not a component in this
  // repo), so its only configurable knob is which of the 4 screen corners
  // it occupies; it can't be reordered, resized, or restyled to match a
  // specific button. bottom-right is the one corner with no app UI in it.
  devIndicators: {
    position: "bottom-right",
  },
  images: {
    remotePatterns: [
      {
        hostname: "avatar.vercel.sh",
      },
      {
        protocol: "https",
        //https://nextjs.org/docs/messages/next-image-unconfigured-host
        hostname: "*.public.blob.vercel-storage.com",
      },
    ],
  },
};

export default nextConfig;
