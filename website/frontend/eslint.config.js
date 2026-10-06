import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import globals from 'globals'

export default tseslint.config(
  { ignores: ['dist/**', 'node_modules/**', 'public/**'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['src/**/*.{ts,tsx}', 'vite.config.ts', 'scripts/**/*.mjs'],
    languageOptions: { globals: { ...globals.browser, ...globals.node } },
    rules: {
      // API responses are gradually being typed; the TypeScript build remains
      // the gate for invalid property access in typed application code.
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
)
