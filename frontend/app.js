fetch("/api/results")                       // 1. ask the backend for the data
  .then(response => response.json())        // 2. turn the answer into a JS object
  .then(data => {                           // 3. use it
    const box = document.getElementById("transcript");
    for (const w of data.windows) {
      box.innerHTML += `<p>${w.startSec}s — attention ${w.attentionLevel} — ${w.text}</p>`;
    }
  });