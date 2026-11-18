/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        serif: ['Newsreader', 'Charter', '"Bitstream Charter"', '"Sitka Text"', 'Cambria', 'serif'],
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        paper: {
          50: '#fdfcf9',
          100: '#f8f6f0',
          200: '#efece1',
          300: '#e3dfd0',
          400: '#ccc7b3',
          500: '#aba48d',
        },
        ink: {
          950: '#0c0e12',
          900: '#151821',
          800: '#232733',
          700: '#394052',
          600: '#525b73',
          500: '#717c99',
          400: '#95a0bd',
          300: '#c0c8de',
          200: '#e2e6f0',
          100: '#f1f3f8',
        },
        scholar: {
          DEFAULT: '#1e3a8a',
          hover: '#172554',
          light: '#eff6ff',
          border: '#bfdbfe',
        }
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(0, 0, 0, 0.04)',
        'card': '0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05)',
        'modal': '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
      }
    },
  },
  plugins: [],
}
