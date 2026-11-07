/**
 * @file app.js
 * @brief Main application orchestrator for the MUDA WebAssembly Playground.
 */

import { CompilerClient } from "./compiler_client.js";
import { CodeEditor } from "./editor.js";
import { KernelInspector } from "./inspector.js";
import { SAMPLES } from "./samples.js";
import { copyToClipboard, decodeShareUrl, downloadFile, encodeShareUrl } from "./utils.js";

class PlaygroundApp {
  constructor() {
    this.compiler = new CompilerClient();
    this.currentMode = "cuda"; // "cuda" or "ptx"
    this.currentSampleId = "vector_add";
    this.lastResult = null;
    this.debounceTimer = null;

    this._initElements();
    this._populateSamples();
    this._initEditor();
    this._initTabs();
    this._bindActions();
    this._loadInitialState();
  }

  _initElements() {
    this.el = {
      editorInput: document.getElementById("editor-input"),
      lineNumbers: document.getElementById("line-numbers"),
      modeSelector: document.getElementById("mode-selector"),
      sampleSelector: document.getElementById("sample-selector"),
      copyBtn: document.getElementById("btn-copy"),
      downloadBtn: document.getElementById("btn-download"),
      shareBtn: document.getElementById("btn-share"),
      mslOutput: document.getElementById("msl-output"),
      mslLineNumbers: document.getElementById("msl-line-numbers"),
      inspectorContainer: document.getElementById("inspector-container"),
      diagnosticsContainer: document.getElementById("diagnostics-container"),
      statusBadge: document.getElementById("status-badge"),
      notification: document.getElementById("notification")
    };

    this.inspector = new KernelInspector(this.el.inspectorContainer);
  }

  _populateSamples() {
    this.el.sampleSelector.innerHTML = "";
    const filtered = SAMPLES.filter(s => s.mode === this.currentMode);

    filtered.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.name}`;
      this.el.sampleSelector.appendChild(opt);
    });

    if (filtered.length > 0) {
      this.currentSampleId = filtered[0].id;
      this.el.sampleSelector.value = this.currentSampleId;
    }
  }

  _initEditor() {
    this.editor = new CodeEditor(
      this.el.editorInput,
      this.el.lineNumbers,
      (code, immediate) => {
        if (immediate) {
          this.triggerCompile();
        } else {
          this._scheduleCompile();
        }
      }
    );
  }

  _initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        tabBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        const target = btn.dataset.tab;
        document.querySelectorAll(".tab-pane").forEach(pane => {
          pane.classList.remove("active");
        });
        const activePane = document.getElementById(`tab-${target}`);
        if (activePane) activePane.classList.add("active");
      });
    });
  }

  _bindActions() {
    this.el.modeSelector.addEventListener("change", (e) => {
      this.currentMode = e.target.value;
      this._populateSamples();
      const sample = SAMPLES.find(s => s.id === this.el.sampleSelector.value);
      if (sample) {
        this.editor.setValue(sample.code);
      }
      this.triggerCompile();
    });

    this.el.sampleSelector.addEventListener("change", (e) => {
      this.currentSampleId = e.target.value;
      const sample = SAMPLES.find(s => s.id === this.currentSampleId);
      if (sample) {
        this.editor.setValue(sample.code);
        this.triggerCompile();
      }
    });

    this.el.copyBtn.addEventListener("click", async () => {
      if (this.lastResult && this.lastResult.msl) {
        await copyToClipboard(this.lastResult.msl);
        this._showNotification("Metal Shading Language (MSL) copied to clipboard!");
      }
    });

    this.el.downloadBtn.addEventListener("click", () => {
      if (this.lastResult && this.lastResult.msl) {
        const filename = `${this.currentSampleId || "kernel"}.metal`;
        downloadFile(filename, this.lastResult.msl);
        this._showNotification(`Downloaded ${filename}`);
      }
    });

    this.el.shareBtn.addEventListener("click", async () => {
      const url = encodeShareUrl(this.editor.getValue(), this.currentMode, this.currentSampleId);
      await copyToClipboard(url);
      this._showNotification("Shareable URL copied to clipboard!");
    });
  }

  _loadInitialState() {
    const shared = decodeShareUrl();
    if (shared && shared.code) {
      if (shared.mode) {
        this.currentMode = shared.mode;
        this.el.modeSelector.value = this.currentMode;
        this._populateSamples();
      }
      this.editor.setValue(shared.code);
      this._showNotification("Loaded shared code from URL");
    } else {
      const sample = SAMPLES.find(s => s.id === this.currentSampleId);
      if (sample) {
        this.editor.setValue(sample.code);
      }
    }
    this.triggerCompile();
  }

  _scheduleCompile() {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.triggerCompile();
    }, 150);
  }

  async triggerCompile() {
    const code = this.editor.getValue();
    if (this.el.statusBadge) {
      this.el.statusBadge.textContent = "Compiling...";
      this.el.statusBadge.className = "status-badge status-compiling";
    }

    const { elapsedMs, result } = await this.compiler.compile(code, this.currentMode);
    this.lastResult = result;

    if (result.ok) {
      if (this.el.statusBadge) {
        this.el.statusBadge.textContent = "Valid MSL";
        this.el.statusBadge.className = "status-badge status-ok";
      }
      if (this.el.mslOutput) {
        this.el.mslOutput.textContent = result.msl;
      }
      this._updateMslLineNumbers(result.msl);

      this.inspector.render(result);

      if (this.el.diagnosticsContainer) {
        this.el.diagnosticsContainer.innerHTML = `
          <div class="diag-banner diag-ok">
            <div class="diag-icon">✓</div>
            <div class="diag-body">
              <strong>Translation Succeeded</strong>
              <p>Generated clean Metal Shading Language for ${result.kernels.length} kernel(s) in ${elapsedMs.toFixed(2)} ms.</p>
            </div>
          </div>
        `;
      }
    } else {
      if (this.el.statusBadge) {
        this.el.statusBadge.textContent = "Error";
        this.el.statusBadge.className = "status-badge status-err";
      }
      const errText = result.msl || "// Compilation error occurred.\n// See Diagnostics tab.";
      if (this.el.mslOutput) {
        this.el.mslOutput.textContent = errText;
      }
      this._updateMslLineNumbers(errText);

      this.inspector.render(null);

      if (this.el.diagnosticsContainer) {
        this.el.diagnosticsContainer.innerHTML = `
          <div class="diag-banner diag-error">
            <div class="diag-icon">✕</div>
            <div class="diag-body">
              <strong>Translation Error [${result.diagnostic_id || "ERR"}]</strong>
              <p class="diag-msg">${result.error || "Syntax or lowering error"}</p>
            </div>
          </div>
        `;
      }
    }
  }

  _updateMslLineNumbers(text) {
    const lines = (text || "").split("\n").length;
    const nums = [];
    for (let i = 1; i <= lines; ++i) {
      nums.push(i);
    }
    this.el.mslLineNumbers.innerHTML = nums.join("<br/>");
  }

  _showNotification(msg) {
    this.el.notification.textContent = msg;
    this.el.notification.classList.add("show");
    setTimeout(() => {
      this.el.notification.classList.remove("show");
    }, 2800);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new PlaygroundApp();
});

