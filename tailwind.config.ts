import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "#121417",
        paper: "#f6f4ef",
        ember: "#ff6b2c",
        moss: "#3f6b4f",
        glacier: "#3a7ca5"
      },
      boxShadow: {
        soft: "0 18px 60px rgba(18, 20, 23, 0.12)",
        deep: "0 24px 80px rgba(0, 0, 0, 0.38)",
        glow: "0 18px 44px rgba(34, 211, 238, 0.24)"
      }
    }
  },
  plugins: []
};

export default config;
