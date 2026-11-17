// What to tell the person about the built-in engine's memory, from `engine_memory`.
//
// Returns `null` when there is nothing to say (the common case, and every model run by another
// engine, which reports nothing), otherwise `{ kind, text, title }`: `kind` is a status-pill
// class, `text` the pill label, `title` the longer explanation.

const MB = 2 ** 20;
const GB = 2 ** 30;

const size = (b) => (b >= GB ? `${(b / GB).toFixed(1)} GB` : `${Math.max(1, Math.round(b / MB))} MB`);

export function describeEngineMemory(m) {
  if (!m) return null;
  if (m.paging) {
    const per = m.pagedBytesPerToken > 0 ? ` (about ${size(m.pagedBytesPerToken)} read for each word)` : "";
    return {
      kind: "bad",
      text: "Too big for free memory",
      title:
        `This model no longer fits in the memory that is free, so it is read from storage again ` +
        `for every word${per} and replies are very slow. Close other apps or pick a smaller model.`,
    };
  }
  if (m.pressureLevel === "warning" || m.pressureLevel === "critical") {
    const freed =
      m.pressureSheds > 0
        ? ` LLMario has freed ${size(m.pressureReleasedBytes)} of saved conversation memory ${m.pressureSheds} time${m.pressureSheds === 1 ? "" : "s"}.`
        : "";
    const relief = m.pressureRelief
      ? " Idle conversations are saved to disk and freed while this lasts."
      : " Freeing idle conversations is turned off.";
    return {
      kind: m.pressureLevel === "critical" ? "bad" : "warn",
      text: m.pressureLevel === "critical" ? "Memory critical" : "Memory low",
      title: `This computer is short of memory.${relief}${freed}`,
    };
  }
  return null;
}
