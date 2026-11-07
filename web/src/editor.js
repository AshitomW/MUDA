/**
 * @file editor.js
 * @brief Lightweight, zero-dependency code editor with line numbers, tab indent, and syntax styling.
 */

export class CodeEditor {
  constructor(textareaEl, lineNumbersEl, onUpdate) {
    this.textarea = textareaEl;
    this.lineNumbers = lineNumbersEl;
    this.onUpdate = onUpdate;

    this._bindEvents();
    this.updateLineNumbers();
  }

  getValue() {
    return this.textarea.value;
  }

  setValue(val) {
    this.textarea.value = val;
    this.updateLineNumbers();
    if (this.onUpdate) this.onUpdate(val);
  }

  updateLineNumbers() {
    const lines = this.textarea.value.split("\n").length;
    const nums = [];
    for (let i = 1; i <= lines; ++i) {
      nums.push(i);
    }
    this.lineNumbers.innerHTML = nums.join("<br/>");
  }

  _bindEvents() {
    this.textarea.addEventListener("input", () => {
      this.updateLineNumbers();
      if (this.onUpdate) this.onUpdate(this.textarea.value);
    });

    this.textarea.addEventListener("scroll", () => {
      this.lineNumbers.scrollTop = this.textarea.scrollTop;
    });

    this.textarea.addEventListener("keydown", (e) => {
      // Support Tab key indentation (2 spaces)
      if (e.key === "Tab") {
        e.preventDefault();
        const start = this.textarea.selectionStart;
        const end = this.textarea.selectionEnd;
        const val = this.textarea.value;

        this.textarea.value = val.substring(0, start) + "  " + val.substring(end);
        this.textarea.selectionStart = this.textarea.selectionEnd = start + 2;
        this.updateLineNumbers();
        if (this.onUpdate) this.onUpdate(this.textarea.value);
      }

      // Cmd+Enter or Ctrl+Enter trigger compilation immediately
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        if (this.onUpdate) this.onUpdate(this.textarea.value, true);
      }
    });
  }
}

