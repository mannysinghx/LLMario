// Run: node --test apps/desktop/ui/markdown.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { renderMarkdown } from "./markdown.js";

test("model output cannot inject markup", () => {
  const evil = [
    '<img src=x onerror="alert(1)">',
    "<script>alert(1)</script>",
    "**<b onmouseover=alert(1)>x</b>**",
    "`<iframe src=javascript:alert(1)>`",
    "[click](javascript:alert(1))",
    '```html"><script>alert(1)</script>\n<script>alert(2)</script>\n```',
    "# <svg onload=alert(1)>",
    "- <a href=x>y</a>",
  ].join("\n\n");
  const html = renderMarkdown(evil);
  assert.doesNotMatch(html, /<(script|img|iframe|svg|a|b)\b/i, html);
  assert.doesNotMatch(html, /<[^>]*\son\w+=/i, "no event-handler attribute inside a real tag");
  assert.match(html, /&lt;script&gt;/);
});

test("links are shown as text, never clickable", () => {
  const html = renderMarkdown("see [docs](https://example.com)");
  assert.doesNotMatch(html, /<a\b/);
  assert.match(html, /docs <span class="md-url">\(https:\/\/example\.com\)<\/span>/);
});

test("common markdown renders", () => {
  const html = renderMarkdown("# Title\n\n**bold** and *it* and `code`\n\n- a\n- b\n\n1. one\n2. two\n\n```py\nprint('hi')\n```");
  assert.match(html, /<h3>Title<\/h3>/);
  assert.match(html, /<strong>bold<\/strong> and <em>it<\/em> and <code>code<\/code>/);
  assert.match(html, /<ul><li>a<\/li><li>b<\/li><\/ul>/);
  assert.match(html, /<ol><li>one<\/li><li>two<\/li><\/ol>/);
  assert.match(html, /<span>py<\/span>/);
  assert.match(html, /print\(&#39;hi&#39;\)/);
});

test("unclosed code fence while streaming still renders safely", () => {
  const html = renderMarkdown("```js\nconst a = '<b>';");
  assert.match(html, /<pre><code>const a = &#39;&lt;b&gt;&#39;;<\/code><\/pre>/);
});

test("fence language is sanitized", () => {
  const html = renderMarkdown('```"onmouseover=x\ncode\n```');
  // It may appear as escaped visible text, never inside a tag.
  assert.doesNotMatch(html, /<[^>]*onmouseover/);
});
