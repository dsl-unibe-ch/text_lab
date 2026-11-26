/*
 * Behaviour of the Text Lab web UI that htmx does not cover.
 *
 * - Live job updates: one server-sent event stream per page, open only
 *   while a job of the session waits or runs (an element with
 *   data-job-active is on the page). Browsers allow only six connections
 *   to the portal for all its tabs, so a page never holds more than one.
 * - The navigation on narrow screens, the job indicator's popover, the
 *   colour theme switch and the session's time left.
 * - Form helpers: drop zones, upload progress, switches that reveal
 *   settings, choices that show parts of a form, range values, tabs.
 * - Error notices for failed htmx requests, copying error details.
 *
 * There are no inline scripts (the Content Security Policy forbids them):
 * everything is wired through data- attributes.
 */
(function () {
  "use strict";

  var root = document.body.getAttribute("data-root") || "";

  /* ---- Formatting ----------------------------------------------------- */

  function fileSize(bytes) {
    if (bytes < 1000) {
      return bytes + " bytes";
    }
    var units = ["kB", "MB", "GB", "TB"];
    var size = bytes;
    for (var index = 0; index < units.length; index += 1) {
      size /= 1000;
      if (size < 1000 || index === units.length - 1) {
        return size.toFixed(1).replace(/\.0$/, "") + " " + units[index];
      }
    }
    return bytes + " bytes";
  }

  function duration(seconds) {
    seconds = Math.max(0, Math.floor(seconds));
    var hours = Math.floor(seconds / 3600);
    var minutes = Math.floor((seconds % 3600) / 60);
    var rest = seconds % 60;
    var pad = function (value) {
      return value < 10 ? "0" + value : String(value);
    };
    if (hours) {
      return hours + " h " + pad(minutes) + " min";
    }
    if (minutes) {
      return minutes + " min " + pad(rest) + " s";
    }
    return rest + " s";
  }

  /* ---- Notices -------------------------------------------------------- */

  var notices = document.getElementById("notices");

  function showError(title, text) {
    if (!notices) {
      return;
    }
    var notice = document.createElement("div");
    notice.className = "notice notice--error";
    notice.setAttribute("role", "alert");
    var label = document.createElement("p");
    label.className = "label notice__label";
    label.textContent = "Error";
    var body = document.createElement("div");
    body.className = "notice__body";
    var heading = document.createElement("p");
    heading.className = "notice__title";
    heading.textContent = title;
    var message = document.createElement("div");
    message.className = "notice__text";
    message.textContent = text;
    var close = document.createElement("button");
    close.className = "icon-button";
    close.type = "button";
    close.setAttribute("data-dismiss", "");
    close.setAttribute("aria-label", "Dismiss");
    close.textContent = "\u00d7";
    body.appendChild(heading);
    body.appendChild(message);
    notice.appendChild(label);
    notice.appendChild(body);
    notice.appendChild(close);
    notices.prepend(notice);
  }

  document.addEventListener("htmx:responseError", function (event) {
    var xhr = event.detail.xhr;
    if (xhr.getResponseHeader("X-Textlab-Notice") && window.htmx) {
      window.htmx.swap(notices, xhr.responseText, { swapStyle: "afterbegin" });
    } else {
      showError(
        "The request failed (status " + xhr.status + ")",
        "Reload the page and try again. If it keeps failing, the session may have ended."
      );
    }
  });

  document.addEventListener("htmx:sendError", function () {
    showError(
      "Text Lab cannot be reached",
      "Check in Open OnDemand that the session still runs."
    );
  });

  /* ---- Live job updates ----------------------------------------------- */

  var stream = null;

  function closeStream() {
    if (stream) {
      stream.close();
      stream = null;
    }
  }

  function openStream() {
    var page = document.body.getAttribute("data-page") || "";
    stream = new EventSource(root + "/events?page=" + encodeURIComponent(page));
    stream.addEventListener("update", function (event) {
      // The data holds out-of-band fragments only: job cards and the job
      // indicator, each replacing the element with its id.
      window.htmx.swap(document.body, event.data, { swapStyle: "none" });
      syncStream();
    });
    stream.addEventListener("idle", closeStream);
    stream.addEventListener("error", function () {
      // The browser reconnects by itself after network errors. A stream it
      // gave up on was refused: the login ended or the server stopped.
      if (stream && stream.readyState === EventSource.CLOSED) {
        closeStream();
        window.location.reload();
      }
    });
  }

  function syncStream() {
    if (!window.htmx || !window.EventSource) {
      return;
    }
    var active = document.querySelector("[data-job-active]") !== null;
    if (active && !stream) {
      openStream();
    } else if (!active && stream) {
      closeStream();
    }
  }

  document.addEventListener("htmx:afterSettle", syncStream);
  window.addEventListener("pagehide", closeStream);

  /* Running times count on between updates. */
  var counters = new WeakMap();
  window.setInterval(function () {
    var now = Date.now();
    document.querySelectorAll("[data-elapsed]").forEach(function (element) {
      if (!counters.has(element)) {
        counters.set(element, now - Number(element.getAttribute("data-elapsed")) * 1000);
      }
      element.textContent = duration((now - counters.get(element)) / 1000);
    });
  }, 1000);

  /* The session's time left, in the status bar; red in the last 15 min. */
  function timeLeft(seconds) {
    var minutes = Math.floor(Math.max(0, seconds) / 60);
    if (minutes < 1) {
      return "under 1 min";
    }
    var hours = Math.floor(minutes / 60);
    minutes %= 60;
    if (hours) {
      return hours + " h " + (minutes < 10 ? "0" : "") + minutes + " min";
    }
    return minutes + " min";
  }

  var session = document.querySelector("[data-session-left]");
  if (session) {
    var sessionEnd = Date.now() + Number(session.getAttribute("data-session-left")) * 1000;
    var countdown = session.querySelector("[data-session-countdown]");
    var showTimeLeft = function () {
      var left = (sessionEnd - Date.now()) / 1000;
      if (countdown) {
        countdown.textContent = timeLeft(left);
      }
      session.classList.toggle("is-ending", left < 15 * 60);
    };
    showTimeLeft();
    window.setInterval(showTimeLeft, 30000);
  }

  /* ---- Navigation, popovers, theme ------------------------------------ */

  function setMenu(open) {
    document.body.classList.toggle("nav-open", open);
    document.querySelectorAll("[data-menu-toggle]").forEach(function (button) {
      button.setAttribute("aria-expanded", String(open));
    });
    var backdrop = document.querySelector("[data-menu-close]");
    if (backdrop) {
      backdrop.hidden = !open;
    }
    if (open) {
      var current = document.querySelector("#sidebar [aria-current='page'], #sidebar a");
      if (current) {
        current.focus();
      }
    }
  }

  function closePopovers(except) {
    document.querySelectorAll("[data-popover]").forEach(function (button) {
      var panel = document.getElementById(button.getAttribute("data-popover"));
      if (panel && panel !== except) {
        panel.hidden = true;
        button.setAttribute("aria-expanded", "false");
      }
    });
  }

  var themes = ["system", "light", "dark"];
  var themeNames = { system: "as the system", light: "light", dark: "dark" };
  var themeShort = { system: "Auto", light: "Light", dark: "Dark" };

  function currentTheme() {
    try {
      return window.localStorage.getItem("textlab-theme") || "system";
    } catch (error) {
      return "system";
    }
  }

  function applyTheme(theme) {
    if (theme === "system") {
      document.documentElement.removeAttribute("data-theme");
    } else {
      document.documentElement.setAttribute("data-theme", theme);
    }
    document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
      var label = "Colour theme: " + themeNames[theme];
      button.setAttribute("data-current", theme);
      button.setAttribute("aria-label", label);
      button.title = label;
      var name = button.querySelector("[data-theme-name]");
      if (name) {
        name.textContent = themeShort[theme];
      }
    });
  }

  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      return;
    }

    if (target.closest("[data-menu-toggle]")) {
      setMenu(!document.body.classList.contains("nav-open"));
      return;
    }
    if (target.closest("[data-menu-close]")) {
      setMenu(false);
      return;
    }

    var popoverButton = target.closest("[data-popover]");
    if (popoverButton) {
      var panel = document.getElementById(popoverButton.getAttribute("data-popover"));
      if (panel) {
        var open = panel.hidden;
        closePopovers(panel);
        panel.hidden = !open;
        popoverButton.setAttribute("aria-expanded", String(open));
      }
      return;
    }
    if (!target.closest(".popover")) {
      closePopovers(null);
    }

    if (target.closest("[data-theme-toggle]")) {
      var next = themes[(themes.indexOf(currentTheme()) + 1) % themes.length];
      try {
        if (next === "system") {
          window.localStorage.removeItem("textlab-theme");
        } else {
          window.localStorage.setItem("textlab-theme", next);
        }
      } catch (error) {
        // The choice then lasts for this page only.
      }
      applyTheme(next);
      return;
    }

    var dismiss = target.closest("[data-dismiss]");
    if (dismiss) {
      var notice = dismiss.closest(".notice");
      if (notice) {
        notice.remove();
      }
      return;
    }

    var copy = target.closest("[data-copy]");
    if (copy) {
      var details = copy.closest("details");
      var text = details ? details.querySelector("pre") : null;
      if (text && navigator.clipboard) {
        navigator.clipboard.writeText(text.textContent).then(function () {
          copy.textContent = "Copied";
        });
      }
      return;
    }

    var tab = target.closest("[role='tab']");
    if (tab) {
      selectTab(tab);
    }
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      setMenu(false);
      closePopovers(null);
      return;
    }
    var tab = event.target instanceof Element ? event.target.closest("[role='tab']") : null;
    if (tab && (event.key === "ArrowRight" || event.key === "ArrowLeft")) {
      var tabs = Array.prototype.slice.call(
        tab.closest("[role='tablist']").querySelectorAll("[role='tab']")
      );
      var step = event.key === "ArrowRight" ? 1 : -1;
      var next = tabs[(tabs.indexOf(tab) + step + tabs.length) % tabs.length];
      selectTab(next);
      next.focus();
      event.preventDefault();
    }
  });

  function selectTab(tab) {
    var list = tab.closest("[role='tablist']");
    list.querySelectorAll("[role='tab']").forEach(function (other) {
      var selected = other === tab;
      other.setAttribute("aria-selected", String(selected));
      other.tabIndex = selected ? 0 : -1;
      var panel = document.getElementById(other.getAttribute("aria-controls"));
      if (panel) {
        panel.hidden = !selected;
      }
    });
  }

  /* ---- Forms ---------------------------------------------------------- */

  function listFiles(input) {
    var zone = input.closest("[data-dropzone]");
    var list = document.querySelector("[data-file-list='" + input.name + "']");
    var errorLine = document.querySelector("[data-file-error='" + input.name + "']");
    var limit = zone ? Number(zone.getAttribute("data-max-bytes")) : 0;
    var accepted = (input.getAttribute("accept") || "")
      .split(",")
      .map(function (item) {
        return item.trim().toLowerCase();
      })
      .filter(Boolean);
    var problem = "";
    if (list) {
      list.textContent = "";
    }
    Array.prototype.forEach.call(input.files || [], function (file) {
      var name = file.name.toLowerCase();
      var typeOk = !accepted.length || accepted.some(function (suffix) {
        return name.endsWith(suffix);
      });
      var sizeOk = !limit || file.size <= limit;
      if (!typeOk) {
        problem = file.name + " is not a type this page reads (" + accepted.join(", ") + ").";
      } else if (!sizeOk) {
        problem = file.name + " is larger than " + fileSize(limit) + ".";
      }
      if (list) {
        var item = document.createElement("li");
        item.className = "file-chip" + (typeOk && sizeOk ? "" : " file-chip--error");
        var label = document.createElement("span");
        label.className = "file-chip__name";
        label.textContent = file.name;
        var size = document.createElement("span");
        size.className = "file-chip__size";
        size.textContent = fileSize(file.size);
        item.appendChild(label);
        item.appendChild(size);
        list.appendChild(item);
      }
    });
    input.setCustomValidity(problem);
    if (errorLine) {
      errorLine.textContent = problem;
      errorLine.hidden = !problem;
    }
  }

  function applySwitch(form) {
    var name = form.getAttribute("data-switch");
    var chosen = form.querySelector("input[name='" + name + "']:checked");
    var value = chosen ? chosen.value : "";
    form.querySelectorAll("[data-when]").forEach(function (part) {
      var shown = part.getAttribute("data-when").split(" ").indexOf(value) !== -1;
      part.hidden = !shown;
      // Controls in hidden parts are neither checked nor sent.
      part.querySelectorAll("input, select, textarea, button").forEach(function (control) {
        control.disabled = !shown;
      });
      if (part.matches("button")) {
        part.disabled = !shown;
      }
    });
  }

  function applyReveal(input) {
    var target = document.getElementById(input.getAttribute("data-reveal"));
    if (target) {
      target.hidden = !input.checked;
    }
  }

  function showRange(input) {
    document.querySelectorAll("[data-output-of='" + input.id + "']").forEach(function (output) {
      output.textContent = input.value;
    });
  }

  document.addEventListener("change", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    if (target.matches("[data-dropzone] input[type='file']")) {
      listFiles(target);
    }
    if (target.matches("[data-reveal]")) {
      applyReveal(target);
    }
    var switchForm = target.closest("[data-switch]");
    if (switchForm && target.name === switchForm.getAttribute("data-switch")) {
      applySwitch(switchForm);
    }
  });

  document.addEventListener("input", function (event) {
    if (event.target instanceof Element && event.target.matches("input[type='range']")) {
      showRange(event.target);
    }
  });

  ["dragenter", "dragover"].forEach(function (type) {
    document.addEventListener(type, function (event) {
      var zone = event.target instanceof Element ? event.target.closest("[data-dropzone]") : null;
      if (zone) {
        zone.classList.add("is-dragover");
      }
    });
  });
  ["dragleave", "drop"].forEach(function (type) {
    document.addEventListener(type, function (event) {
      var zone = event.target instanceof Element ? event.target.closest("[data-dropzone]") : null;
      if (zone) {
        zone.classList.remove("is-dragover");
      }
    });
  });

  /* Upload progress: a form with [data-upload-progress] inside shows it. */
  document.addEventListener("htmx:xhr:progress", function (event) {
    var form = event.target instanceof Element ? event.target.closest("form") : null;
    var box = form ? form.querySelector("[data-upload-progress]") : null;
    if (!box || !event.detail.lengthComputable || !event.detail.total) {
      return;
    }
    var share = event.detail.loaded / event.detail.total;
    box.hidden = share >= 1;
    box.querySelector("progress").value = share;
    box.querySelector("[data-upload-progress-text]").textContent =
      "Uploading: " + fileSize(event.detail.loaded) + " of " + fileSize(event.detail.total);
  });

  document.addEventListener("htmx:afterRequest", function (event) {
    var form = event.target instanceof Element ? event.target.closest("form") : null;
    var box = form ? form.querySelector("[data-upload-progress]") : null;
    if (box) {
      box.hidden = true;
    }
    if (form && event.detail.successful && form.hasAttribute("data-upload-form")) {
      form.reset();
      form.querySelectorAll("[data-dropzone] input[type='file']").forEach(listFiles);
    }
  });

  /* ---- Start ---------------------------------------------------------- */

  function initialise(scope) {
    scope.querySelectorAll("[data-reveal]").forEach(applyReveal);
    scope.querySelectorAll("[data-switch]").forEach(applySwitch);
    scope.querySelectorAll("input[type='range']").forEach(showRange);
  }

  applyTheme(currentTheme());
  initialise(document);
  if (window.htmx) {
    window.htmx.onLoad(function (element) {
      if (element instanceof Element && element !== document.body) {
        initialise(element);
      }
    });
  }
  syncStream();
})();
