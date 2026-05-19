const dropzone     = document.getElementById("dropzone");
const fileInput    = document.getElementById("file-input");
const preview      = document.getElementById("preview");
const uploadPrompt = document.getElementById("upload-prompt");
const predictBtn   = document.getElementById("predict-btn");
const loading      = document.getElementById("loading");
const results      = document.getElementById("results");

let selectedFile = null;

// ── mobile samples drawer ──────────────────────────────────────────
const samplesToggle   = document.getElementById("samples-toggle");
const samplesPanel    = document.getElementById("samples-panel");
const samplesBackdrop = document.getElementById("samples-backdrop");
const samplesClose    = document.getElementById("samples-close");

function openSamplesDrawer() {
  samplesPanel.classList.remove("translate-x-full");
  samplesPanel.classList.add("translate-x-0");
  samplesBackdrop.classList.remove("hidden");
}

function closeSamplesDrawer() {
  samplesPanel.classList.add("translate-x-full");
  samplesPanel.classList.remove("translate-x-0");
  samplesBackdrop.classList.add("hidden");
}

samplesToggle.addEventListener("click", openSamplesDrawer);
samplesClose.addEventListener("click", closeSamplesDrawer);
samplesBackdrop.addEventListener("click", closeSamplesDrawer);

// ── sample sidebar ─────────────────────────────────────────────────
const sampleList = document.getElementById("sample-list");

(async () => {
  const files = await fetch("pics-list").then((r) => r.json());
  files.forEach((filename) => {
    const imgUrl = `pics/${filename}`;
    const btn = document.createElement("button");
    btn.className =
      "w-full flex items-center gap-2 p-2 rounded-xl text-left hover:bg-green-50 transition-colors";
    btn.innerHTML = `
      <img src="${imgUrl}" class="w-full aspect-square object-cover rounded-lg" alt="${filename}" />
    `;
    btn.addEventListener("click", async () => {
      closeSamplesDrawer();
      const res = await fetch(imgUrl);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type });
      setFile(file);
      runPredict();
    });
    sampleList.appendChild(btn);
  });
})();

// ── label formatting ───────────────────────────────────────────────
function formatLabel(raw) {
  return raw.replace(/___/g, " — ").replace(/_/g, " ");
}

// ── file selection ─────────────────────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("border-green-400", "bg-green-50");
});
dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("border-green-400", "bg-green-50");
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("border-green-400", "bg-green-50");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadPrompt.classList.add("hidden");
    preview.src = e.target.result;
    preview.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
  predictBtn.disabled = false;
  results.classList.add("hidden");
}

// ── predict ────────────────────────────────────────────────────────
predictBtn.addEventListener("click", runPredict);

async function runPredict() {
  if (!selectedFile) return;

  loading.classList.remove("hidden");
  results.classList.add("hidden");
  predictBtn.disabled = true;

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch("predict", { method: "POST", body: form });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg);
    }
    displayResults(await res.json());
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    loading.classList.add("hidden");
    predictBtn.disabled = false;
  }
}

// ── render results ─────────────────────────────────────────────────
function displayResults(data) {
  // Predicted class + confidence bar
  document.getElementById("predicted-class").textContent =
    formatLabel(data.predicted_class);

  const pct = Math.round(data.confidence * 100);
  document.getElementById("confidence-text").textContent = pct + "%";

  const bar = document.getElementById("confidence-bar");
  bar.style.width = pct + "%";
  bar.className =
    "h-2.5 rounded-full transition-all duration-700 " +
    (pct > 70 ? "bg-green-500" : pct > 40 ? "bg-yellow-400" : "bg-red-400");

  document.getElementById("confidence-warning").classList.toggle("hidden", pct >= 40);

  // Top-K list
  const list = document.getElementById("top-k-list");
  list.innerHTML = "";
  data.top_k.forEach((item) => {
    const p = Math.round(item.score * 100);
    list.insertAdjacentHTML(
      "beforeend",
      `<div>
        <div class="flex justify-between text-sm mb-1">
          <span class="text-gray-700 font-medium">${formatLabel(item.label)}</span>
          <span class="text-gray-400">${p}%</span>
        </div>
        <div class="w-full bg-gray-100 rounded-full h-1.5">
          <div class="h-1.5 rounded-full bg-emerald-400" style="width:${p}%"></div>
        </div>
      </div>`
    );
  });

  // Similar images
  const grid    = document.getElementById("similar-images");
  const section = document.getElementById("similar-section");
  grid.innerHTML = "";

  if (data.retrieved_examples && data.retrieved_examples.length > 0) {
    section.classList.remove("hidden");
    data.retrieved_examples.forEach((path) => {
      grid.insertAdjacentHTML(
        "beforeend",
        `<img src="${path}"
              class="w-full aspect-square object-cover rounded-lg shadow-sm"
              onerror="this.parentElement.removeChild(this)"
              alt="similar plant" />`
      );
    });
  } else {
    section.classList.add("hidden");
  }

  results.classList.remove("hidden");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}
