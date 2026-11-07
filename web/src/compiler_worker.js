/**
 * @file compiler_worker.js
 * @brief Dedicated Web Worker hosting the MUDA WebAssembly compilation engine.
 */

let mudaModule = null;
let isInitializing = false;
let initPromise = null;

async function initModule() {
  if (mudaModule) return mudaModule;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      // In web worker context, load the Emscripten module script
      importScripts("../dist/muda_compiler.js");
      if (typeof createMudaModule === "function") {
        mudaModule = await createMudaModule({
          locateFile: (path) => `../dist/${path}`
        });
        return mudaModule;
      }
    } catch (err) {
      console.warn("Wasm engine unavailable in current context:", err);
    }
    return null;
  })();

  return initPromise;
}

self.onmessage = async (e) => {
  const { id, mode, source } = e.data;
  const t0 = performance.now();

  try {
    const mod = await initModule();
    if (!mod) {
      // Fallback response if wasm binary is not yet compiled
      self.postMessage({
        id,
        elapsedMs: performance.now() - t0,
        result: {
          ok: false,
          code: 999,
          diagnostic_id: "WASM_NOT_COMPILED",
          error: "Wasm runtime not yet compiled. Run 'web/wasm/build_wasm.sh' or deploy with GitHub Pages to generate muda_compiler.wasm.",
          msl: "// WebAssembly runtime not compiled.\n// Run web/wasm/build_wasm.sh to build muda_compiler.wasm.",
          kernels: []
        }
      });
      return;
    }

    let result;
    if (mode === "ptx") {
      result = mod.translatePtx(source);
    } else {
      result = mod.translateCuda(source);
    }

    const elapsedMs = performance.now() - t0;
    self.postMessage({
      id,
      elapsedMs,
      result
    });
  } catch (err) {
    self.postMessage({
      id,
      elapsedMs: performance.now() - t0,
      result: {
        ok: false,
        code: -1,
        diagnostic_id: "WASM_EXCEPTION",
        error: String(err && err.message ? err.message : err),
        msl: "",
        kernels: []
      }
    });
  }
};

