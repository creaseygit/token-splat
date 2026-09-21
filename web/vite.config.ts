import { defineConfig } from "vite";

export default defineConfig({
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
