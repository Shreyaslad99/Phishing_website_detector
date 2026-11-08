// content/content.js
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "SHOW_RESULT") {
    const { verdict, score } = msg.payload;
    const banner = document.createElement("div");
    banner.style.position = "fixed";
    banner.style.bottom = "10px";
    banner.style.right = "10px";
    banner.style.zIndex = "2147483647";
    banner.style.background = verdict === "phishing" ? "#ff4a4a" : "#4caf50";
    banner.style.color = "#fff";
    banner.style.padding = "6px 10px";
    banner.style.borderRadius = "4px";
    banner.style.fontSize = "13px";
    banner.textContent = `🛡 ${verdict} (${(score*100).toFixed(1)} %)`;
    document.body.appendChild(banner);
    setTimeout(() => banner.remove(), 8000);
  }
});
