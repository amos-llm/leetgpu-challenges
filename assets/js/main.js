(function () {
  "use strict";

  /* ============================================================
     Theme
     ============================================================ */
  function getPreferredTheme() {
    const stored = localStorage.getItem("theme");
    if (stored === "dark" || stored === "light") return stored;
    return window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);

    const isDark = theme === "dark";
    const darkSheet = document.getElementById("prism-dark");
    const lightSheet = document.getElementById("prism-light");
    if (darkSheet) darkSheet.media = isDark ? "all" : "none";
    if (lightSheet) lightSheet.media = isDark ? "none" : "all";
  }

  const themeToggle = document.getElementById("theme-toggle");
  if (themeToggle) {
    setTheme(getPreferredTheme());
    themeToggle.addEventListener("click", function () {
      const current = document.documentElement.dataset.theme;
      setTheme(current === "dark" ? "light" : "dark");
    });
  }

  /* ============================================================
     Filter tabs (index page)
     ============================================================ */
  const filterTabs = document.querySelectorAll(".filter-tab");
  const frameworkTabs = document.querySelectorAll(".framework-tab");
  const grid = document.getElementById("challenge-grid");
  const searchInput = document.getElementById("search");
  const emptyState = document.getElementById("empty-state");

  if (filterTabs.length && grid) {
    function readHash() {
      var h = window.location.hash.replace(/^#/, "");
      var m = {};
      if (h) {
        h.split("&").forEach(function (part) {
          var kv = part.split("=");
          if (kv.length === 2) m[kv[0]] = decodeURIComponent(kv[1]);
        });
      }
      return m;
    }

    function writeHash(difficulty, framework, query) {
      var parts = [];
      if (difficulty && difficulty !== "all") parts.push("difficulty=" + difficulty);
      if (framework && framework !== "all") parts.push("framework=" + framework);
      if (query) parts.push("q=" + encodeURIComponent(query));
      var h = parts.length ? "#" + parts.join("&") : "";
      if (window.location.hash !== h) {
        history.replaceState(null, "", h || window.location.pathname);
      }
    }

    function applyFilter() {
      var activeFilter = document.querySelector(".filter-tab.active");
      var difficulty = activeFilter ? activeFilter.dataset.filter : "all";
      var activeFramework = document.querySelector(".framework-tab.active");
      var framework = activeFramework ? activeFramework.dataset.framework : "all";
      var query = searchInput ? searchInput.value.toLowerCase().trim() : "";

      var cards = grid.querySelectorAll(".challenge-card");
      var visible = 0;
      cards.forEach(function (card) {
        var cardDifficulty = card.dataset.difficulty;
        var name = card.dataset.name || "";
        var number = card.dataset.number || "";
        var cardFrameworks = (card.dataset.frameworks || "").split(",");

        var okDifficulty = difficulty === "all" || cardDifficulty === difficulty;
        var okFramework = framework === "all" || cardFrameworks.indexOf(framework) !== -1;
        var okSearch =
          !query ||
          name.indexOf(query) !== -1 ||
          number === query ||
          number.toString() === query;

        var ok = okDifficulty && okFramework && okSearch;
        card.classList.toggle("hidden", !ok);
        if (ok) visible++;
      });

      if (emptyState) {
        emptyState.hidden = visible > 0;
      }

      writeHash(difficulty, framework, query);
    }

    function activateTab(tabs, attr, value) {
      tabs.forEach(function (t) { t.classList.remove("active"); });
      var target = document.querySelector('[' + attr + '="' + value + '"]');
      if (target) target.classList.add("active");
      else if (tabs.length) tabs[0].classList.add("active");
    }

    filterTabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        activateTab(filterTabs, 'data-filter', tab.dataset.filter);
        applyFilter();
      });
    });

    if (frameworkTabs.length) {
      frameworkTabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
          activateTab(frameworkTabs, 'data-framework', tab.dataset.framework);
          applyFilter();
        });
      });
    }

    // Restore state from URL hash on load
    var hashState = readHash();
    if (hashState.difficulty) {
      activateTab(filterTabs, 'data-filter', hashState.difficulty);
    }
    if (hashState.framework) {
      activateTab(frameworkTabs, 'data-framework', hashState.framework);
    }
    if (hashState.q && searchInput) {
      searchInput.value = hashState.q;
    }

    applyFilter();

    if (searchInput) {
      var searchTimer;
      searchInput.addEventListener("input", function () {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(applyFilter, 150);
      });
    }

    // Handle back/forward
    window.addEventListener("popstate", function () {
      var s = readHash();
      if (s.difficulty) activateTab(filterTabs, 'data-filter', s.difficulty);
      else activateTab(filterTabs, 'data-filter', 'all');
      if (s.framework) activateTab(frameworkTabs, 'data-framework', s.framework);
      else if (frameworkTabs.length) activateTab(frameworkTabs, 'data-framework', 'all');
      if (searchInput) searchInput.value = s.q || "";
      applyFilter();
    });
  }

  /* ============================================================
     Code tabs (detail page)
     ============================================================ */
  const codeTabsGroups = document.querySelectorAll(".code-tabs");
  codeTabsGroups.forEach(function (group) {
    const buttons = group.querySelectorAll(".tab-btn");
    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        // Deactivate all in this group
        buttons.forEach(function (b) {
          b.classList.remove("active");
        });
        group.querySelectorAll(".tab-panel").forEach(function (p) {
          p.classList.remove("active");
        });

        // Activate target
        btn.classList.add("active");
        const targetId = btn.dataset.tab;
        const panel = group.querySelector(
          '[data-panel="' + targetId + '"]'
        );
        if (panel) {
          panel.classList.add("active");
          // Highlight newly visible code block
          if (typeof Prism !== "undefined") {
            const code = panel.querySelector("code[class*='language-']");
            if (code) Prism.highlightElement(code);
          }
        }
      });
    });
  });

  /* ============================================================
     Copy code buttons
     ============================================================ */
  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var wrapper = btn.closest(".code-block-wrapper");
      if (!wrapper) return;
      var code = wrapper.querySelector("code");
      if (!code) return;

      var text = code.textContent;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          btn.classList.add("copied");
          setTimeout(function () {
            btn.classList.remove("copied");
          }, 2000);
        });
      } else {
        var textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        try {
          document.execCommand("copy");
          btn.classList.add("copied");
          setTimeout(function () {
            btn.classList.remove("copied");
          }, 2000);
        } catch (err) { }
        document.body.removeChild(textarea);
      }
    });
  });
})();
