/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND_URL: string;
  readonly MODE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// Add process.env support for Node.js/Docker environments
declare namespace NodeJS {
  interface ProcessEnv {
    readonly VITE_BACKEND_URL?: string;
    readonly MODE?: string;
    readonly NODE_ENV?: string;
  }
}

// Ensure process is available in the global scope
declare const process: {
  env: NodeJS.ProcessEnv;
};
