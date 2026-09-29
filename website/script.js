// Copy buttons for code blocks, simple tabs, and Windows tabs for Windows visitors. No tracking,
// no external requests.
document.querySelectorAll(".code .copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const text = btn.parentElement.querySelector("code").textContent;
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Select & copy";
    }
    setTimeout(() => (btn.textContent = "Copy"), 1600);
  });
});

document.querySelectorAll("[data-tabs]").forEach((tabs) => {
  const buttons = tabs.querySelectorAll('[role="tab"]');
  buttons.forEach((b) =>
    b.addEventListener("click", () => {
      buttons.forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      tabs.querySelectorAll("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== b.dataset.tab));
    })
  );
});

// Windows visitors see the Windows setup tabs first. Nothing is stored or sent anywhere.
if (/Windows/.test(navigator.userAgent)) {
  document.querySelectorAll('[role="tab"][data-os="windows"]').forEach((b) => b.click());
}
