/**
 * @file utils.js
 * @brief Utilities for URL hash sharing, file download, and clipboard management.
 */

export function encodeShareUrl(code, mode, sampleId) {
  try {
    const payload = JSON.stringify({ code, mode, sampleId });
    const b64 = btoa(encodeURIComponent(payload));
    window.location.hash = `share=${b64}`;
    return window.location.href;
  } catch (e) {
    console.error("Failed to encode state:", e);
    return window.location.href;
  }
}

export function decodeShareUrl() {
  try {
    const hash = window.location.hash;
    if (!hash || !hash.includes("share=")) return null;
    const b64 = hash.split("share=")[1];
    if (!b64) return null;
    const payload = decodeURIComponent(atob(b64));
    return JSON.parse(payload);
  } catch (e) {
    console.warn("Could not decode shared URL hash:", e);
    return null;
  }
}

export async function copyToClipboard(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(text);
    return true;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
  return true;
}

export function downloadFile(filename, content, mimeType = "text/plain") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

