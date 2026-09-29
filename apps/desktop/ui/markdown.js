// Minimal, safe Markdown renderer for model output.
//
// Safety: every piece of model text is HTML-escaped before any markup is added, and the only
// markup produced is a fixed set of tags with static attributes. Links are shown as text (never
// clickable) so model output cannot navigate the app window.

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ESC[c]);

function inline(text) {
  const codes = [];
  let s = escapeHtml(text).replace(/`([^`\n]+)`/g, (_, c) => {
    codes.push(c);
    return `\u0000${codes.length - 1}\u0000`;
  });
  s = s
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?=[^*\w]|$)/g, "$1<em>$2</em>")
    .replace(/(^|[^_\w])_([^_\n]+)_(?=[^_\w]|$)/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
    .replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, '$1 <span class="md-url">($2)</span>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
}

export function renderMarkdown(src) {
  const lines = (src || "").replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let para = [];
  let list = null; // { tag, items }

  const flushPara = () => {
    if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`);
    para = [];
  };
  const flushList = () => {
    if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`);
    list = null;
  };
  const flush = () => {
    flushPara();
    flushList();
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s*```\s*([\w+#.-]*)\s*$/);
    if (fence) {
      flush();
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) code.push(lines[i++]);
      const lang = fence[1] ? escapeHtml(fence[1]) : "";
      out.push(
        `<div class="md-code"><div class="md-code-head"><span>${lang || "code"}</span>` +
          `<button class="md-copy" type="button">Copy</button></div>` +
          `<pre><code>${escapeHtml(code.join("\n"))}</code></pre></div>`
      );
      continue;
    }
    let m;
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
      flush();
      const n = Math.min(m[1].length + 2, 6); // keep headings modest inside chat
      out.push(`<h${n}>${inline(m[2])}</h${n}>`);
    } else if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flush();
      out.push("<hr>");
    } else if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) {
      flushPara();
      if (!list || list.tag !== "ul") {
        flushList();
        list = { tag: "ul", items: [] };
      }
      list.items.push(m[1]);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      flushPara();
      if (!list || list.tag !== "ol") {
        flushList();
        list = { tag: "ol", items: [] };
      }
      list.items.push(m[1]);
    } else if ((m = line.match(/^>\s?(.*)$/))) {
      flush();
      out.push(`<blockquote>${inline(m[1])}</blockquote>`);
    } else if (!line.trim()) {
      flush();
    } else {
      flushList();
      para.push(line);
    }
  }
  flush();
  return out.join("");
}
