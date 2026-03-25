module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        primary: "#2563EB",   // Blue 600
        secondary: "#60A5FA", // Blue 400
        background: "#F8FAFC", // Slate 50
        surface: "#ffffff",
      },
      animation: {
        aurora: "aurora 60s linear infinite",
        shimmer: "shimmer 2s linear infinite",
        spotlight: "spotlight 2s ease .75s 1 forwards",
      },
      keyframes: {
        spotlight: {
          "0%": { opacity: 0, transform: "translate(-72%, -62%) scale(0.5)" },
          "100%": { opacity: 1, transform: "translate(-50%, -40%) scale(1)" },
        },
        aurora: {
          from: { backgroundPosition: "50% 50%, 50% 50%" },
          to: { backgroundPosition: "350% 50%, 350% 50%" },
        },
        shimmer: {
          from: { backgroundPosition: "0 0" },
          to: { backgroundPosition: "-200% 0" },
        },
      },
    },
  },
  plugins: [],
};
