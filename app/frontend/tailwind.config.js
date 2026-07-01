/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        ink: "#172033",
        ocean: "#0f766e",
        signal: "#2563eb",
        coral: "#e11d48",
        sun: "#d97706",
      },
      boxShadow: {
        panel: "0 18px 50px rgba(15, 23, 42, 0.08)",
      },
    },
  },
  plugins: [],
};
