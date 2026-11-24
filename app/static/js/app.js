/* HealthPredict UI helpers: theme, form UX, gauges. No dependencies besides Bootstrap. */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  // ---- theme toggle ----
  const root = document.documentElement;
  const btn = $("#themeToggle");
  const paint = () => { if (btn) btn.innerHTML = root.getAttribute("data-bs-theme") === "dark"
      ? '<i class="bi bi-sun"></i>' : '<i class="bi bi-moon-stars"></i>'; };
  paint();
  btn && btn.addEventListener("click", () => {
    const next = root.getAttribute("data-bs-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-bs-theme", next);
    try { localStorage.setItem("hp-theme", next); } catch (e) {}
    paint();
    document.dispatchEvent(new CustomEvent("hp-theme", { detail: next }));
  });

  // ---- auto-dismiss toasts ----
  $$(".toast-stack .alert").forEach((el, i) => setTimeout(() => {
    try { bootstrap.Alert.getOrCreateInstance(el).close(); } catch (e) {}
  }, 6000 + i * 800));

  // ---- loading state on submit ----
  $$("form[data-loading]").forEach((f) => f.addEventListener("submit", () => {
    const b = f.querySelector("button[type=submit]");
    if (b) { b.disabled = true; b.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>' + f.dataset.loading; }
  }));

  // ---- character counter + example texts ----
  const ta = $("textarea[data-counter]");
  if (ta) {
    const out = $("#" + ta.dataset.counter);
    const upd = () => { const n = ta.value.length; out.textContent = n + " / " + ta.maxLength;
      out.classList.toggle("text-danger", n > 0 && n < 15); };
    ta.addEventListener("input", upd); upd();
    $$("[data-example]").forEach((b) => b.addEventListener("click", () => {
      ta.value = b.dataset.example; ta.focus(); upd();
    }));
  }

  // ---- BMI calculator ----
  $$("[data-bmi-target]").forEach((box) => {
    const h = $("input[name=_height]", box), w = $("input[name=_weight]", box), res = $(".bmi-result", box);
    const target = document.getElementById(box.dataset.bmiTarget);
    const calc = () => {
      const hm = parseFloat(h.value) / 100, kg = parseFloat(w.value);
      if (hm > 0.5 && kg > 10) { const bmi = kg / (hm * hm); res.textContent = "BMI " + bmi.toFixed(1);
        target.value = bmi.toFixed(1); target.dispatchEvent(new Event("input")); }
      else { res.textContent = ""; }
    };
    h.addEventListener("input", calc); w.addEventListener("input", calc);
  });

  // ---- password visibility ----
  $$("[data-toggle-pw]").forEach((b) => b.addEventListener("click", () => {
    const inp = document.getElementById(b.dataset.togglePw);
    inp.type = inp.type === "password" ? "text" : "password";
    b.innerHTML = inp.type === "password" ? '<i class="bi bi-eye"></i>' : '<i class="bi bi-eye-slash"></i>';
  }));

  // ---- animate gauges and bars in ----
  requestAnimationFrame(() => setTimeout(() => {
    $$(".gauge .bar[data-offset]").forEach((p) => p.style.strokeDashoffset = p.dataset.offset);
    $$("[data-width]").forEach((el) => el.style.width = el.dataset.width);
  }, 60));
})();
