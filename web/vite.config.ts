import { defineConfig } from "vite";

// Deployed under https://creaseygit.github.io/token-splat/. Vite's `base`
// makes both the JS bundle URLs and `import.meta.env.BASE_URL` reflect that.
export default defineConfig({
  base: process.env.NODE_ENV === "production" ? "/token-splat/" : "/",
  server: { port: 5173, strictPort: false },
  build: {
    target: "es2022",
    sourcemap: true,
    // Assets are streamed by hand (kf_XX.bin), so keep chunking modest.
    rollupOptions: { output: { manualChunks: undefined } },
  },
  // onnxruntime-web ships wasm/JSEP files that must be copied verbatim.
  assetsInclude: ["**/*.wasm", "**/*.onnx"],
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@huggingface/transformers"],
  },
});
