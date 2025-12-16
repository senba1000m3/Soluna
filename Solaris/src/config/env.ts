/**
 * Environment Configuration
 * Supports both import.meta.env (Vite) and process.env (Node.js/Docker)
 */

/**
 * Get environment variable with fallback support
 * Checks import.meta.env first (Vite), then process.env (Node.js)
 */
function getEnv(key: string, defaultValue: string = ''): string {
  // Check import.meta.env (Vite)
  if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env[key]) {
    return import.meta.env[key] as string;
  }

  // Check process.env (Node.js/Docker)
  if (typeof process !== 'undefined' && process.env && process.env[key]) {
    return process.env[key] as string;
  }

  // Return default value
  return defaultValue;
}

/**
 * Application Configuration
 */
export const config = {
  /**
   * Backend API URL
   * Defaults to /api for Docker/Caddy setup, or http://localhost:8000 for local dev
   */
  backendUrl: getEnv('VITE_BACKEND_URL', '/api'),

  /**
   * Environment mode (development, production, test)
   */
  mode: getEnv('MODE', 'production'),

  /**
   * Check if running in development mode
   */
  isDev: getEnv('MODE', 'production') === 'development',

  /**
   * Check if running in production mode
   */
  isProd: getEnv('MODE', 'production') === 'production',
} as const;

/**
 * Export individual values for convenience
 */
export const BACKEND_URL = config.backendUrl;
export const IS_DEV = config.isDev;
export const IS_PROD = config.isProd;

/**
 * Default export
 */
export default config;
