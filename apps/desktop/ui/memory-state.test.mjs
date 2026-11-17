// Run: node --test apps/desktop/ui/memory-state.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { describeEngineMemory } from "./memory-state.js";

const calm = {
  pressureLevel: "normal",
  pressureRelief: true,
  pressureSheds: 0,
  pressureReleasedBytes: 0,
  paging: false,
  pagedBytesPerToken: 0,
};

test("nothing to say when memory is fine or the engine reports nothing", () => {
  assert.equal(describeEngineMemory(null), null);
  assert.equal(describeEngineMemory(undefined), null);
  assert.equal(describeEngineMemory(calm), null);
  assert.equal(describeEngineMemory({ ...calm, pressureLevel: "unknown" }), null);
});

test("a model read from storage for every word is the loudest state", () => {
  const d = describeEngineMemory({ ...calm, paging: true, pagedBytesPerToken: 6.5 * 2 ** 30, pressureLevel: "critical" });
  assert.equal(d.kind, "bad");
  assert.match(d.title, /read from storage again/);
  assert.match(d.title, /6\.5 GB/);
});

test("pressure shows what was freed, and says when freeing is off", () => {
  const w = describeEngineMemory({ ...calm, pressureLevel: "warning", pressureSheds: 1, pressureReleasedBytes: 351 * 2 ** 20 });
  assert.equal(w.kind, "warn");
  assert.match(w.title, /351 MB/);
  assert.match(w.title, /1 time\./);
  const c = describeEngineMemory({ ...calm, pressureLevel: "critical", pressureRelief: false });
  assert.equal(c.kind, "bad");
  assert.match(c.title, /turned off/);
});

test("the label is plain text built only from fixed strings and numbers", () => {
  const d = describeEngineMemory({ ...calm, pressureLevel: "warning" });
  assert.equal(d.text, "Memory low");
});
