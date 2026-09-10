import { defineConfig } from "vite";

// `npm run dev` proxies /api to the FastAPI server (run separately with
// `uvicorn server:app --reload`). `npm run build` emits dist/, which
// server.py serves directly so the whole app can run as one process.
export default defineConfig({
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
