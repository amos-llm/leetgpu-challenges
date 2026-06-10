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
  const grid = document.getElementById("challenge-grid");
  const searchInput = document.getElementById("search");

  if (filterTabs.length && grid) {
    function applyFilter() {
      const activeFilter = document.querySelector(".filter-tab.active");
      const difficulty = activeFilter ? activeFilter.dataset.filter : "all";
      const query = searchInput ? searchInput.value.toLowerCase().trim() : "";

      const cards = grid.querySelectorAll(".challenge-card");
      cards.forEach(function (card) {
        const cardDifficulty = card.dataset.difficulty;
        const name = card.dataset.name || "";
        const number = card.dataset.number || "";

        const matchesFilter =
          difficulty === "all" || cardDifficulty === difficulty;
        const matchesSearch =
          !query ||
          name.includes(query) ||
          number === query ||
          number.toString() === query;

        card.classList.toggle("hidden", !(matchesFilter && matchesSearch));
      });
    }

    filterTabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        filterTabs.forEach(function (t) {
          t.classList.remove("active");
        });
        tab.classList.add("active");
        applyFilter();
      });
    });

    // Check URL for ?difficulty= param (from breadcrumb links)
    const params = new URLSearchParams(window.location.search);
    const diffParam = params.get("difficulty");
    if (diffParam) {
      const targetTab = document.querySelector(
        '.filter-tab[data-filter="' + diffParam + '"]'
      );
      if (targetTab) {
        filterTabs.forEach(function (t) {
          t.classList.remove("active");
        });
        targetTab.classList.add("active");
      }
    }

    if (searchInput) {
      searchInput.addEventListener("input", applyFilter);
    }

    /* ============================================================
       Framework filter tabs (index page)
       ============================================================ */
    const frameworkTabs = document.querySelectorAll(".framework-tab");

    if (frameworkTabs.length) {
      // Extend applyFilter to include framework filtering
      applyFilter = function () {
        const activeFilter = document.querySelector(".filter-tab.active");
        const difficulty = activeFilter ? activeFilter.dataset.filter : "all";
        const activeFramework = document.querySelector(".framework-tab.active");
        const framework = activeFramework
          ? activeFramework.dataset.framework
          : "all";
        const query = searchInput
          ? searchInput.value.toLowerCase().trim()
          : "";

        const cards = grid.querySelectorAll(".challenge-card");
        cards.forEach(function (card) {
          const cardDifficulty = card.dataset.difficulty;
          const name = card.dataset.name || "";
          const number = card.dataset.number || "";
          const cardFrameworks = (card.dataset.frameworks || "").split(",");

          const matchesDifficulty =
            difficulty === "all" || cardDifficulty === difficulty;
          const matchesFramework =
            framework === "all" ||
            cardFrameworks.indexOf(framework) !== -1;
          const matchesSearch =
            !query ||
            name.indexOf(query) !== -1 ||
            number === query ||
            number.toString() === query;

          card.classList.toggle(
            "hidden",
            !(matchesDifficulty && matchesFramework && matchesSearch)
          );
        });
      };

      frameworkTabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
          frameworkTabs.forEach(function (t) {
            t.classList.remove("active");
          });
          tab.classList.add("active");
          applyFilter();
        });
      });
    }
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
