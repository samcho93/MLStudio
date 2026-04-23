/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'node-data': '#3b82f6',
        'node-layer': '#8b5cf6',
        'node-training': '#f59e0b',
        'node-viz': '#10b981',
        'node-output': '#ef4444',
      },
    },
  },
  plugins: [],
};
