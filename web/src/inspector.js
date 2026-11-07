/**
 * @file inspector.js
 * @brief Renders kernel signature reflection and hardware Metal buffer index mappings.
 */

export class KernelInspector {
  constructor(containerEl) {
    this.container = containerEl;
  }

  render(result) {
    if (!result || !result.ok || !result.kernels || result.kernels.length === 0) {
      this.container.innerHTML = `
        <div class="empty-inspector">
          <p class="text-muted">No active kernel reflection available. Compile a valid CUDA or PTX kernel to inspect Metal buffer bindings.</p>
        </div>
      `;
      return;
    }

    const html = result.kernels.map((kernel, kIdx) => {
      const paramsHtml = (kernel.parameters && kernel.parameters.length > 0)
        ? `
          <table class="reflection-table">
            <thead>
              <tr>
                <th>Param</th>
                <th>Kind</th>
                <th>Metal Type</th>
                <th>Buffer Index</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              ${kernel.parameters.map(p => `
                <tr>
                  <td><code>${p.name}</code></td>
                  <td><span class="badge ${p.is_pointer ? 'badge-ptr' : 'badge-val'}">${p.is_pointer ? (p.is_const ? 'const ptr*' : 'ptr*') : 'value'}</span></td>
                  <td><code>${p.metal_type}</code></td>
                  <td><span class="buffer-tag">[[buffer(${p.buffer_index})]]</span></td>
                  <td>${p.byte_size} B</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        `
        : `<p class="text-muted">No explicit parameters.</p>`;

      const sharedMem = kernel.static_shared_memory_bytes || 0;
      const dynamicShared = kernel.uses_dynamic_shared_memory ? "Yes" : "No";

      return `
        <div class="kernel-card">
          <div class="kernel-header">
            <span class="kernel-title">Kernel #${kIdx + 1}: <strong>${kernel.name}</strong></span>
            <span class="msl-symbol">MSL Symbol: <code>${kernel.msl_name}</code></span>
          </div>

          <div class="kernel-memory-stats">
            <div class="stat-pill">
              <span class="stat-label">Threadgroup Shared Memory:</span>
              <span class="stat-value">${sharedMem} Bytes</span>
            </div>
            <div class="stat-pill">
              <span class="stat-label">Dynamic Shared Memory:</span>
              <span class="stat-value">${dynamicShared}</span>
            </div>
          </div>

          <div class="kernel-params-wrapper">
            <h4 class="params-heading">Parameter Bindings & Address Spaces</h4>
            ${paramsHtml}
          </div>
        </div>
      `;
    }).join("");

    this.container.innerHTML = html;
  }
}

