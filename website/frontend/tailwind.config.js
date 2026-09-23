/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        serif: ['Instrument Serif', 'ui-serif', 'Georgia', 'serif'],
      },
      colors: {
        paper: '#F7F5F3',
        ink: '#37322F',
        surface: '#FFFFFF',
        // Warm neutral scale shared by landing, forms, and dashboard.
        slate: {
          50: '#F7F5F3',
          100: '#F0EDEB',
          200: '#E0DEDB',
          300: '#CEC9C5',
          400: '#9E9792',
          500: '#77716C',
          600: '#605A57',
          700: '#49423D',
          800: '#3E3935',
          900: '#37322F',
        },
        primary: {
          50: '#F4F1EF',
          100: '#E8E2DE',
          200: '#D9D0CA',
          300: '#BEB2AA',
          400: '#8C8078',
          500: '#605A57',
          600: '#37322F',
          700: '#322D2B',
          800: '#292522',
          900: '#211D1B',
        },
        teal: {
          50: '#EFF4F0',
          100: '#DCE9DE',
          200: '#BED4C2',
          300: '#9ABAA0',
          400: '#719878',
          500: '#52775A',
          600: '#3E6548',
          700: '#31523A',
          800: '#284330',
          900: '#203727',
        },
        risk: {
          low: '#3E6548',
          medium: '#B07A37',
          high: '#B45342',
        },
      },
    },
  },
  plugins: [],
}
