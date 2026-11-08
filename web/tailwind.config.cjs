/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{tsx,ts,js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0a0a0a",
        panel: "#161616",
        border: "#2a2a2a",
        text: "#d4d4d4",
        muted: "#9a9a9a",
        accent: "#e11d2e",
      },
      fontFamily: {
        sans: ["'Inter'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};
