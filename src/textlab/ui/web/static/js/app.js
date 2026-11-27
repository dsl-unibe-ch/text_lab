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
 *   settings, choices that show parts of a form, range values, readouts
 *   that follow a choice, tabs.
 * - The audio player of transcripts: plays the recording, marks the
 *   passage and the word being spoken, jumps to a passage or word.
 * - Citations in a summary without a recording show their line in the
 *   transcript.
 * - The translation editor: counts, clearing, Ctrl+Enter, translating as
 *   you type, swapping the languages, copying, the glossary's rows.
 * - OCR results: one page at a time, and the regions outlined on a page
 *   linked to their text.
 * - Chat: Enter sends, the message box grows, Send turns into Stop while a
 *   reply is written, files dropped on the page are attached, the reply
 *   being written stays in view.
 * - The data agents' results, in Chat and Visualize Data: their charts are
 *   drawn with plotly.js, loaded only then; code can be copied; boxes the
 *   reader opened or closed stay so when their card is replaced. A card
 *   the page has in the same version is not replaced at all.
 * - Check boxes ticked or cleared together, with their count, such as the
 *   columns of a table to analyze.
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

  var noticeLabels = { info: "Note", warning: "Warning", error: "Error" };

  function showError(title, text) {
    showNotice("error", title, text);
  }

  function showNotice(level, title, text) {
    if (!notices) {
      return;
    }
    var notice = document.createElement("div");
    notice.className = "notice notice--" + level;
    notice.setAttribute("role", level === "error" ? "alert" : "status");
    var label = document.createElement("p");
    label.className = "label notice__label";
    label.textContent = noticeLabels[level] || "Note";
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
      var chat = chatBeforeUpdate();
      window.htmx.swap(document.body, event.data, { swapStyle: "none" });
      chatAfterUpdate(chat);
      syncStream();
      showDetected();
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

  document.addEventListener("htmx:afterSettle", function () {
    syncStream();
    showDetected();
    syncComposer();
  });
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
    document.dispatchEvent(new CustomEvent("textlab:theme"));
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
    var list = document.querySelector("[data-file-list='" + input.id + "']");
    var errorLine = document.querySelector("[data-file-error='" + input.id + "']");
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

  /* A switch ([data-switch], naming a radio group) shows its [data-when]
     parts that list the chosen value. Switches can be nested: a part
     belongs to the closest switch around it. */
  function applySwitch(form) {
    var name = form.getAttribute("data-switch");
    var chosen = form.querySelector("input[name='" + name + "']:checked");
    var value = chosen ? chosen.value : "";
    form.querySelectorAll("[data-when]").forEach(function (part) {
      if (part.parentElement.closest("[data-switch]") !== form) {
        return;
      }
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
    // Showing a part enabled every control in it, also those of the
    // parts a switch inside it hides; let that switch hide them again.
    form.querySelectorAll("[data-switch]").forEach(function (inner) {
      if (inner.closest("[hidden]") === null) {
        applySwitch(inner);
      }
    });
  }

  function applyReveal(input) {
    var target = document.getElementById(input.getAttribute("data-reveal"));
    if (target) {
      target.hidden = !input.checked;
    }
  }

  /* A select with data-readout shows its option's data-readout-text in
     the element with that id, such as the model a language uses; a radio
     button with data-readout shows its own, when it is checked. */
  function showReadout(control) {
    var readout = document.getElementById(control.getAttribute("data-readout"));
    var source = control.matches("select") ? control.options[control.selectedIndex] : control;
    if (!readout || !source || (control.type === "radio" && !control.checked)) {
      return;
    }
    readout.textContent = source.getAttribute("data-readout-text") || "";
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
    if (target.matches("select[data-readout], input[type='radio'][data-readout]")) {
      showReadout(target);
    }
    var switchForm = target.closest("[data-switch]");
    if (switchForm && target.name === switchForm.getAttribute("data-switch")) {
      applySwitch(switchForm);
    }
    if (target.matches("input[type='checkbox'][name]")) {
      countChecks(target.closest("form") || document, target.name);
    }
  });

  /* A group of check boxes of one name: buttons with data-check="all" or
     "none" and data-check-name tick or clear them all, and an element with
     data-check-count="name" says how many of them are ticked. */
  function checkBoxes(scope, name) {
    return Array.prototype.filter.call(
      scope.querySelectorAll("input[type='checkbox']"),
      function (box) {
        return box.name === name;
      }
    );
  }

  function countChecks(scope, name) {
    var readout = Array.prototype.find.call(scope.querySelectorAll("[data-check-count]"), function (element) {
      return element.getAttribute("data-check-count") === name;
    });
    if (!readout) {
      return;
    }
    var boxes = checkBoxes(scope, name);
    var ticked = boxes.filter(function (box) {
      return box.checked;
    }).length;
    readout.textContent = ticked + " of " + boxes.length;
  }

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
    var sent = event.detail.elt === form || (event.detail.elt && event.detail.elt.hasAttribute("hx-encoding"));
    if (form && event.detail.successful && form.hasAttribute("data-upload-form") && sent) {
      // The files are sent; the settings stay for the next one.
      form.querySelectorAll("[data-dropzone] input[type='file']").forEach(function (input) {
        input.value = "";
        listFiles(input);
      });
    }
  });

  /* ---- Audio player --------------------------------------------------- */

  /*
   * A player ([data-player], with data-duration) holds an <audio
   * data-player-audio>, a play button, the time, a waveform and the
   * transcript: passages (.turn) and words (.w) with their times in data-s
   * and data-e, in order. Its state is built when it is first used. The
   * recording loads only then (preload="none"), so a page of results
   * loads no audio.
   */
  var players = new WeakMap();

  function mediaTime(seconds) {
    seconds = Math.max(0, Math.floor(seconds));
    var hours = Math.floor(seconds / 3600);
    var minutes = Math.floor((seconds % 3600) / 60);
    var rest = seconds % 60;
    var pad = function (value) {
      return value < 10 ? "0" + value : String(value);
    };
    return (hours ? hours + ":" : "") + pad(minutes) + ":" + pad(rest);
  }

  function timed(box, selector) {
    var elements = Array.prototype.slice.call(box.querySelectorAll(selector));
    return {
      elements: elements,
      starts: elements.map(function (element) {
        return Number(element.getAttribute("data-s"));
      }),
      ends: elements.map(function (element) {
        return Number(element.getAttribute("data-e"));
      })
    };
  }

  /* The index of the last item that starts at or before time, or -1. */
  function lastStarted(list, time) {
    var low = 0;
    var high = list.starts.length - 1;
    var found = -1;
    while (low <= high) {
      var middle = (low + high) >> 1;
      if (list.starts[middle] <= time) {
        found = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return found;
  }

  function playerOf(element) {
    var box = element.closest("[data-player]");
    if (!box) {
      return null;
    }
    var player = players.get(box);
    if (player) {
      return player;
    }
    var audio = box.querySelector("[data-player-audio]");
    player = {
      box: box,
      audio: audio,
      duration: Number(box.getAttribute("data-duration")) || 0,
      now: box.querySelector("[data-player-now]"),
      toggle: box.querySelector("[data-player-toggle]"),
      wave: box.querySelector("[data-player-wave]"),
      bars: Array.prototype.slice.call(box.querySelectorAll(".waveform__bar")),
      transcript: box.querySelector("[data-player-transcript]"),
      turns: timed(box, ".turn"),
      words: timed(box, ".w"),
      played: -1,
      bar: null,
      turn: null,
      word: null,
      frame: 0
    };
    players.set(box, player);

    var playing = function (on) {
      box.classList.toggle("is-playing", on);
      if (player.toggle) {
        player.toggle.setAttribute("aria-label", on ? "Pause" : "Play");
      }
    };
    audio.addEventListener("play", function () {
      playing(true);
      run(player);
    });
    audio.addEventListener("pause", function () {
      playing(false);
      update(player);
    });
    audio.addEventListener("seeked", function () {
      update(player);
    });
    audio.addEventListener("loadedmetadata", function () {
      if (isFinite(audio.duration) && audio.duration > 0) {
        player.duration = audio.duration;
      }
    });
    audio.addEventListener("error", function () {
      playing(false);
      showError(
        "The recording cannot be played",
        "It may have been removed with its job, or the session ended. Reload the page."
      );
    });
    return player;
  }

  function mark(player, key, element, className) {
    if (player[key] === element) {
      return false;
    }
    if (player[key]) {
      player[key].classList.remove(className);
    }
    if (element) {
      element.classList.add(className);
    }
    player[key] = element;
    return true;
  }

  function update(player) {
    var time = player.audio.currentTime || 0;
    if (player.now) {
      player.now.textContent = mediaTime(time);
    }
    if (player.wave) {
      player.wave.setAttribute("aria-valuenow", String(Math.floor(time)));
      player.wave.setAttribute("aria-valuetext", mediaTime(time));
    }

    var count = player.bars.length;
    if (count && player.duration) {
      var here = Math.min(count - 1, Math.floor((time / player.duration) * count));
      if (here !== player.played) {
        player.bars.forEach(function (bar, index) {
          bar.classList.toggle("is-played", index < here);
        });
        player.played = here;
      }
      mark(player, "bar", player.bars[here], "is-here");
    }

    // A word stays marked through short pauses, a passage through longer.
    var index = lastStarted(player.words, time);
    var word = index >= 0 && time <= player.words.ends[index] + 1 ? player.words.elements[index] : null;
    mark(player, "word", word, "is-current");
    index = lastStarted(player.turns, time);
    var turn = index >= 0 && time <= player.turns.ends[index] + 3 ? player.turns.elements[index] : null;
    if (mark(player, "turn", turn, "turn--current") && turn && !player.audio.paused) {
      follow(player, turn);
    }
  }

  /* Keep the passage being spoken in view in the transcript's box. */
  function follow(player, turn) {
    var list = player.transcript;
    if (!list || list.scrollHeight <= list.clientHeight) {
      return;
    }
    var top = turn.offsetTop;
    if (top < list.scrollTop || top + turn.offsetHeight > list.scrollTop + list.clientHeight) {
      list.scrollTop = Math.max(0, top - list.clientHeight / 4);
    }
  }

  function run(player) {
    window.cancelAnimationFrame(player.frame);
    var step = function () {
      update(player);
      if (!player.audio.paused) {
        player.frame = window.requestAnimationFrame(step);
      }
    };
    step();
  }

  function play(player) {
    var started = player.audio.play();
    if (started && started.catch) {
      // Failures to load show through the audio's error event.
      started.catch(function () {});
    }
  }

  function seek(player, time, andPlay) {
    var end = player.duration || time;
    player.audio.currentTime = Math.max(0, Math.min(time, end));
    update(player);
    if (andPlay && player.audio.paused) {
      play(player);
    }
  }

  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!(target instanceof Element) || !target.closest("[data-player]")) {
      return;
    }
    var player = playerOf(target);
    if (target.closest("[data-player-toggle]")) {
      if (player.audio.paused) {
        play(player);
      } else {
        player.audio.pause();
      }
      return;
    }
    var from = target.closest("[data-seek]") || target.closest(".w");
    if (from) {
      seek(player, Number(from.getAttribute("data-seek") || from.getAttribute("data-s")), true);
    }
  });

  /* Pressing on the waveform jumps there; dragging moves along it. */
  document.addEventListener("pointerdown", function (event) {
    var wave = event.target instanceof Element ? event.target.closest("[data-player-wave]") : null;
    if (!wave || event.button !== 0) {
      return;
    }
    var player = playerOf(wave);
    var seekTo = function (clientX) {
      var box = wave.getBoundingClientRect();
      var share = Math.min(1, Math.max(0, (clientX - box.left) / box.width));
      seek(player, share * player.duration, false);
    };
    var move = function (moved) {
      seekTo(moved.clientX);
    };
    var stop = function () {
      wave.removeEventListener("pointermove", move);
      wave.removeEventListener("pointerup", stop);
      wave.removeEventListener("pointercancel", stop);
    };
    seekTo(event.clientX);
    wave.setPointerCapture(event.pointerId);
    wave.addEventListener("pointermove", move);
    wave.addEventListener("pointerup", stop);
    wave.addEventListener("pointercancel", stop);
  });

  /* The waveform is a slider: arrows move 5 s, page keys 30 s. */
  document.addEventListener("keydown", function (event) {
    var wave = event.target instanceof Element ? event.target.closest("[data-player-wave]") : null;
    if (!wave) {
      return;
    }
    var player = playerOf(wave);
    var now = player.audio.currentTime || 0;
    var moves = {
      ArrowLeft: now - 5,
      ArrowDown: now - 5,
      ArrowRight: now + 5,
      ArrowUp: now + 5,
      PageDown: now - 30,
      PageUp: now + 30,
      Home: 0,
      End: player.duration
    };
    if (Object.prototype.hasOwnProperty.call(moves, event.key)) {
      seek(player, moves[event.key], false);
      event.preventDefault();
    } else if (event.key === " " || event.key === "Enter") {
      if (player.audio.paused) {
        play(player);
      } else {
        player.audio.pause();
      }
      event.preventDefault();
    }
  });

  /* Passages or words: the words view shows each word's time. */
  document.addEventListener("change", function (event) {
    var target = event.target;
    if (!(target instanceof Element) || !target.matches("[data-player-view]")) {
      return;
    }
    var box = target.closest("[data-player]");
    var list = box ? box.querySelector("[data-player-transcript]") : null;
    if (list) {
      list.classList.toggle("transcript--words", target.value === "words");
    }
  });

  /* One recording plays at a time. */
  document.addEventListener(
    "play",
    function (event) {
      document.querySelectorAll("[data-player-audio]").forEach(function (audio) {
        if (audio !== event.target) {
          audio.pause();
        }
      });
    },
    true
  );

  /* ---- Citations ------------------------------------------------------ */

  /* A citation with data-goto shows the transcript line with that
     data-line in the same result: its tab is selected, the line scrolled
     into view and marked for a moment. (Citations with a recording play
     it instead, through data-seek.) */
  var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  document.addEventListener("click", function (event) {
    var link = event.target instanceof Element ? event.target.closest("[data-goto]") : null;
    var result = link ? link.closest(".result") : null;
    if (!result) {
      return;
    }
    var line = result.querySelector("[data-line='" + link.getAttribute("data-goto") + "']");
    if (!line) {
      return;
    }
    var panel = line.closest("[role='tabpanel']");
    if (panel && panel.hidden) {
      var tab = result.querySelector("[role='tab'][aria-controls='" + panel.id + "']");
      if (tab) {
        selectTab(tab);
      }
    }
    line.scrollIntoView({ block: "center", behavior: calm ? "auto" : "smooth" });
    line.classList.add("is-target");
    window.setTimeout(function () {
      line.classList.remove("is-target");
    }, 2000);
  });

  /* ---- Translation editor ------------------------------------------- */

  /*
   * The form [data-translator] holds the source text ([data-text-source]),
   * its Translate button ([data-text-submit]) and the output pane
   * (#text-output), whose card the server replaces as the translation
   * waits, runs and ends. A finished card carries data-detected (the
   * detected language's code) and data-detected-name.
   */
  var liveDelay = 900;
  var liveTimer = 0;
  var livePending = false;

  function translator() {
    return document.querySelector("[data-translator]");
  }

  function textSubmit(form) {
    return form ? form.querySelector("[data-text-submit]") : null;
  }

  function outputCard() {
    return document.querySelector("#text-output .output");
  }

  /* Translate now, or as soon as the request in flight has ended. */
  function translateNow() {
    var form = translator();
    var button = textSubmit(form);
    var source = form ? form.querySelector("[data-text-source]") : null;
    if (!button || !source || button.closest("[hidden]") || !source.value.trim()) {
      return;
    }
    if (button.disabled) {
      livePending = true;
      return;
    }
    livePending = false;
    button.click();
  }

  function liveOn() {
    var toggle = document.querySelector("[data-live-toggle]");
    return toggle !== null && toggle.checked;
  }

  function scheduleLive() {
    window.clearTimeout(liveTimer);
    if (liveOn()) {
      liveTimer = window.setTimeout(translateNow, liveDelay);
    }
  }

  function showCount(area) {
    var count = document.getElementById(area.getAttribute("data-count"));
    if (!count) {
      return;
    }
    var limit = Number(count.getAttribute("data-limit")) || 0;
    var length = area.value.length;
    count.textContent = length.toLocaleString("en") + " / " + limit.toLocaleString("en");
    count.classList.toggle("is-near", limit > 0 && length >= limit * 0.9 && length < limit);
    count.classList.toggle("is-full", limit > 0 && length >= limit);
  }

  /* "Detect language" names the language found for the last translation. */
  function showDetected() {
    var select = document.querySelector("[data-source-language]");
    var option = select ? select.querySelector("option[value='auto']") : null;
    if (!option) {
      return;
    }
    var card = outputCard();
    var name = card ? card.getAttribute("data-detected-name") : "";
    option.textContent = name ? "Detect language (" + name + ")" : "Detect language";
  }

  function glossaryRows(box) {
    return Array.prototype.slice.call(box.querySelectorAll(".glossary__row"));
  }

  function countGlossary(details) {
    var box = details.querySelector("[data-glossary-rows]");
    var out = details.querySelector("[data-glossary-count]");
    if (!box || !out) {
      return;
    }
    var terms = glossaryRows(box).filter(function (row) {
      var inputs = row.querySelectorAll("input");
      return inputs[0].value.trim() && inputs[1].value.trim();
    }).length;
    out.textContent = terms ? terms + (terms === 1 ? " term" : " terms") : "No terms";
    var add = details.querySelector("[data-glossary-add]");
    if (add) {
      add.disabled = glossaryRows(box).length >= Number(box.getAttribute("data-glossary-limit") || 50);
    }
  }

  function addGlossaryRow(details) {
    var box = details.querySelector("[data-glossary-rows]");
    var template = details.querySelector("[data-glossary-template]");
    if (!box || !template) {
      return null;
    }
    var row = template.content.firstElementChild.cloneNode(true);
    box.appendChild(row);
    countGlossary(details);
    row.querySelector("input").focus();
    return row;
  }

  function swapLanguages(form) {
    var source = form.querySelector("[data-source-language]");
    var target = form.querySelector("[data-target-language]");
    var card = outputCard();
    var from = source.value;
    if (from === "auto") {
      from = card ? card.getAttribute("data-detected") || "" : "";
    }
    if (!from) {
      showNotice(
        "info",
        "The language to swap is not known yet",
        "Choose it under From, or translate the text first so that it is detected."
      );
      source.focus();
      return;
    }
    source.value = target.value;
    target.value = from;
    showDetected();
    var text = card ? card.querySelector("[data-output-text]") : null;
    var area = form.querySelector("[data-text-source]");
    if (text && area && !area.closest("[hidden]")) {
      // Translate the translation back.
      area.value = text.textContent;
      showCount(area);
      translateNow();
    }
  }

  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    var form = target.closest("[data-translator]");
    if (target.closest("[data-swap-languages]") && form) {
      swapLanguages(form);
      return;
    }
    if (target.closest("[data-clear-text]") && form) {
      var area = form.querySelector("[data-text-source]");
      area.value = "";
      showCount(area);
      area.focus();
      return;
    }
    var copy = target.closest("[data-copy-from]");
    if (copy) {
      var from = document.getElementById(copy.getAttribute("data-copy-from"));
      if (from && navigator.clipboard) {
        var label = copy.innerHTML;
        navigator.clipboard.writeText(from.textContent).then(
          function () {
            copy.textContent = "Copied";
            window.setTimeout(function () {
              copy.innerHTML = label;
            }, 1500);
          },
          function () {
            showError("The text could not be copied", "Select it and copy it yourself.");
          }
        );
      }
      return;
    }
    var details = target.closest("[data-glossary]");
    if (details && target.closest("[data-glossary-add]")) {
      addGlossaryRow(details);
      return;
    }
    var remove = details ? target.closest("[data-glossary-remove]") : null;
    if (remove) {
      var row = remove.closest(".glossary__row");
      if (glossaryRows(row.parentElement).length > 1) {
        var next = row.nextElementSibling || row.previousElementSibling;
        row.remove();
        if (next && next.querySelector("input")) {
          next.querySelector("input").focus();
        }
      } else {
        row.querySelectorAll("input").forEach(function (input) {
          input.value = "";
        });
        row.querySelector("input").focus();
      }
      countGlossary(details);
    }
  });

  document.addEventListener("keydown", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && target.matches("[data-text-source]")) {
      window.clearTimeout(liveTimer);
      translateNow();
      event.preventDefault();
      return;
    }
    // In the glossary, Enter goes on to the next field instead of sending.
    var row = event.key === "Enter" ? target.closest(".glossary__row") : null;
    if (row && target.matches("input")) {
      event.preventDefault();
      var inputs = row.querySelectorAll("input");
      if (target === inputs[0]) {
        inputs[1].focus();
      } else if (row.nextElementSibling && row.nextElementSibling.querySelector("input")) {
        row.nextElementSibling.querySelector("input").focus();
      } else {
        addGlossaryRow(row.closest("[data-glossary]"));
      }
    }
  });

  document.addEventListener("input", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    if (target.matches("[data-text-source]")) {
      showCount(target);
      scheduleLive();
    }
    var details = target.closest("[data-glossary]");
    if (details) {
      countGlossary(details);
    }
  });

  document.addEventListener("change", function (event) {
    var target = event.target;
    if (!(target instanceof Element) || !target.closest("[data-translator]")) {
      return;
    }
    if (target.matches("[data-live-toggle]")) {
      try {
        window.localStorage.setItem("textlab-live-translation", target.checked ? "on" : "off");
      } catch (error) {
        // The choice then lasts for this page only.
      }
      if (target.checked) {
        translateNow();
      }
      return;
    }
    if (target.matches("[data-source-language]")) {
      showDetected();
    }
    // Another language or engine: translate again when typing does.
    if (target.matches("select, input[type='radio']") && target.name !== "mode") {
      scheduleLive();
    }
  });

  /* A translation that was asked for while one was being sent. */
  document.addEventListener("htmx:afterRequest", function (event) {
    if (livePending && event.detail.elt && event.detail.elt.matches("[data-text-submit]")) {
      window.setTimeout(translateNow, 0);
    }
  });

  function initialiseTranslator(scope) {
    scope.querySelectorAll("[data-text-source]").forEach(showCount);
    scope.querySelectorAll("[data-glossary]").forEach(countGlossary);
    var toggle = scope.querySelector("[data-live-toggle]");
    if (toggle) {
      try {
        toggle.checked = window.localStorage.getItem("textlab-live-translation") === "on";
      } catch (error) {
        toggle.checked = false;
      }
    }
    showDetected();
  }

  /* ---- OCR results ---------------------------------------------------- */

  /*
   * A pager ([data-pager]) shows one of its [data-pager-page] sections,
   * chosen with [data-pager-select] or the [data-pager-step] buttons.
   * Elements with the same data-region (an outline on the page image and
   * the block of its text) are marked together while one is pointed at;
   * clicking an outline brings its text into view. A data-fill button
   * fills a field (the Hugging Face tool's model name).
   */
  function showPage(pager, index) {
    var pages = pager.querySelectorAll("[data-pager-page]");
    index = Math.max(0, Math.min(pages.length - 1, index));
    pages.forEach(function (page) {
      page.hidden = Number(page.getAttribute("data-pager-page")) !== index;
    });
    var select = pager.querySelector("[data-pager-select]");
    if (select) {
      select.value = String(index);
    }
    pager.querySelectorAll("[data-pager-step]").forEach(function (button) {
      var step = Number(button.getAttribute("data-pager-step"));
      button.disabled = (step < 0 && index === 0) || (step > 0 && index === pages.length - 1);
    });
  }

  function linkRegion(element, on) {
    var result = element.closest(".result") || document;
    var id = element.getAttribute("data-region");
    result.querySelectorAll("[data-region='" + id + "']").forEach(function (other) {
      other.classList.toggle("is-linked", on);
    });
  }

  document.addEventListener("change", function (event) {
    var target = event.target;
    if (target instanceof Element && target.matches("[data-pager-select]")) {
      showPage(target.closest("[data-pager]"), Number(target.value));
    }
  });

  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    var step = target.closest("[data-pager-step]");
    if (step) {
      var pager = step.closest("[data-pager]");
      var select = pager.querySelector("[data-pager-select]");
      showPage(pager, Number(select ? select.value : 0) + Number(step.getAttribute("data-pager-step")));
      return;
    }
    var box = target.closest(".box[data-region]");
    if (box) {
      var page = box.closest("[data-pager-page], [role='tabpanel']") || document;
      var block = page.querySelector("[data-region='" + box.getAttribute("data-region") + "']:not(.box)");
      if (block) {
        block.scrollIntoView({ block: "nearest", behavior: calm ? "auto" : "smooth" });
      }
      return;
    }
    var check = target.closest("[data-check]");
    if (check) {
      var group = check.closest("form") || document;
      var name = check.getAttribute("data-check-name");
      checkBoxes(group, name).forEach(function (input) {
        input.checked = check.getAttribute("data-check") === "all";
      });
      countChecks(group, name);
      return;
    }
    // A button with data-fill puts its data-fill-value into the field with
    // that id, such as a downloaded model's name into the model field.
    var fill = target.closest("[data-fill]");
    if (fill) {
      var field = document.getElementById(fill.getAttribute("data-fill"));
      if (field) {
        field.value = fill.getAttribute("data-fill-value") || "";
        field.dispatchEvent(new Event("input", { bubbles: true }));
        field.focus();
      }
    }
  });

  document.addEventListener("mouseover", function (event) {
    var region = event.target instanceof Element ? event.target.closest("[data-region]") : null;
    if (region) {
      linkRegion(region, true);
    }
  });

  document.addEventListener("mouseout", function (event) {
    var region = event.target instanceof Element ? event.target.closest("[data-region]") : null;
    if (region) {
      linkRegion(region, false);
    }
  });

  /* ---- Topic models: pyLDAvis's view ---------------------------------- */

  /*
   * An LDA result shows pyLDAvis's view ([data-ldavis]): a map of topic
   * bubbles and, beside it, a list of terms ([data-ldavis-list]): the most
   * salient ones until a topic is chosen, then the chosen topic's. A topic
   * is chosen by clicking its bubble, the list or the arrows. Its terms
   * rank by relevance, lambda * log probability + (1 - lambda) * log lift,
   * re-ranked when lambda changes; the first ones show, with their bars
   * scaled to the largest overall frequency shown. Pointing at a term sizes
   * the bubbles by how its occurrences spread over the topics
   * (data-shares, "index:share" pairs), as pyLDAvis does.
   */
  function ldavisChoose(root, index) {
    var select = root.querySelector("[data-ldavis-select]");
    var caption = root.querySelector("[data-ldavis-caption]");
    root.querySelectorAll("[data-ldavis-list]").forEach(function (list) {
      var shown = list.getAttribute("data-ldavis-list") === index;
      list.hidden = !shown;
      if (shown && caption) {
        caption.textContent = list.getAttribute("data-caption");
      }
    });
    root.querySelectorAll("[data-ldavis-topic]").forEach(function (topic) {
      topic.classList.toggle("is-chosen", topic.getAttribute("data-ldavis-topic") === index);
    });
    if (select) {
      select.value = index;
    }
    ldavisRank(root);
  }

  function ldavisRank(root) {
    var select = root.querySelector("[data-ldavis-select]");
    var slider = root.querySelector("[data-ldavis-lambda]");
    var list = select && select.value !== ""
      ? root.querySelector("[data-ldavis-list='" + select.value + "']")
      : null;
    if (!list || !slider) {
      return;
    }
    var lambda = Number(slider.value);
    var count = Number(root.getAttribute("data-ldavis-terms")) || 30;
    var rows = Array.prototype.slice.call(list.children).map(function (row) {
      return {
        row: row,
        relevance: lambda * Number(row.getAttribute("data-logprob")) +
          (1 - lambda) * Number(row.getAttribute("data-loglift")),
        total: Number(row.getAttribute("data-total")),
        freq: Number(row.getAttribute("data-freq"))
      };
    });
    rows.sort(function (a, b) {
      return b.relevance - a.relevance;
    });
    var largest = rows.slice(0, count).reduce(function (most, item) {
      return Math.max(most, item.total);
    }, 0) || 1;
    rows.forEach(function (item, place) {
      item.row.hidden = place >= count;
      var total = item.row.querySelector(".ldavis__total");
      var freq = item.row.querySelector(".ldavis__freq");
      if (total) {
        total.setAttribute("width", (100 * item.total / largest).toFixed(2));
      }
      if (freq) {
        freq.setAttribute("width", (100 * item.freq / largest).toFixed(2));
      }
      list.appendChild(item.row);
    });
  }

  function ldavisSpread(root, row) {
    var radius = Number(root.getAttribute("data-ldavis-radius")) || 16;
    var shares = {};
    if (row) {
      (row.getAttribute("data-shares") || "").split(" ").forEach(function (pair) {
        var parts = pair.split(":");
        if (parts.length === 2) {
          shares[parts[0]] = Number(parts[1]);
        }
      });
    }
    root.querySelectorAll("[data-ldavis-topic]").forEach(function (topic) {
      var circle = topic.querySelector("circle");
      var share = shares[topic.getAttribute("data-ldavis-topic")] || 0;
      circle.setAttribute("r", row ? (radius * Math.sqrt(share)).toFixed(2) : circle.getAttribute("data-r"));
    });
  }

  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!(target instanceof Element)) {
      return;
    }
    var root = target.closest("[data-ldavis]");
    if (!root) {
      return;
    }
    var topic = target.closest("[data-ldavis-topic]");
    if (topic) {
      var index = topic.getAttribute("data-ldavis-topic");
      var select = root.querySelector("[data-ldavis-select]");
      ldavisChoose(root, select && select.value === index ? "" : index);
      return;
    }
    var step = target.closest("[data-ldavis-step]");
    if (step) {
      var options = Array.prototype.slice.call(root.querySelector("[data-ldavis-select]").options);
      var current = options.findIndex(function (option) {
        return option.selected;
      });
      var next = (current + Number(step.getAttribute("data-ldavis-step")) + options.length) % options.length;
      ldavisChoose(root, options[next].value);
    }
  });

  document.addEventListener("keydown", function (event) {
    var target = event.target;
    if (!(target instanceof Element) || (event.key !== "Enter" && event.key !== " ")) {
      return;
    }
    var topic = target.closest("[data-ldavis-topic]");
    var root = topic ? topic.closest("[data-ldavis]") : null;
    if (root) {
      event.preventDefault();
      ldavisChoose(root, topic.getAttribute("data-ldavis-topic"));
    }
  });

  document.addEventListener("change", function (event) {
    var target = event.target;
    if (target instanceof Element && target.matches("[data-ldavis-select]")) {
      ldavisChoose(target.closest("[data-ldavis]"), target.value);
    }
  });

  document.addEventListener("input", function (event) {
    var target = event.target;
    if (target instanceof Element && target.matches("[data-ldavis-lambda]")) {
      ldavisRank(target.closest("[data-ldavis]"));
    }
  });

  document.addEventListener("mouseover", function (event) {
    var row = event.target instanceof Element ? event.target.closest(".ldavis__row") : null;
    var root = row ? row.closest("[data-ldavis]") : null;
    if (root) {
      ldavisSpread(root, row);
    }
  });

  document.addEventListener("mouseout", function (event) {
    var row = event.target instanceof Element ? event.target.closest(".ldavis__row") : null;
    var root = row ? row.closest("[data-ldavis]") : null;
    if (root && !(event.relatedTarget instanceof Element && row.contains(event.relatedTarget))) {
      ldavisSpread(root, null);
    }
  });

  /* ---- Chat ----------------------------------------------------------- */

  /*
   * The page [data-chat] holds the conversation (#chat-thread: a message
   * and its reply's card per .exchange), the message box (form
   * [data-chat-composer]) and the side panel (#chat-panel-body, refreshed
   * from data-chat-panel-url after each reply). A reply's card carries
   * data-job-active while it waits or is written.
   *
   * The page follows the reply being written: while the reader is at the
   * foot of the conversation, every change of its size (a word, the
   * thinking, a chart drawn) scrolls to the foot again (a ResizeObserver,
   * so it does not depend on what changed it). Scrolling up stops that;
   * scrolling back to the foot, or sending a message, starts it again.
   * Once no reply is written (and its charts are drawn), the page stays
   * put, so a box the reader opens does not move it.
   */
  var keptOpen = {};
  var plotlyLoading = null;
  var chatFollow = true;
  var chatLastY = window.scrollY;
  var followUntil = 0;
  var touchY = null;

  // Milliseconds the page still follows a reply after it ended: its
  // charts are drawn then.
  var FOLLOW_AFTER = 4000;

  function chatPage() {
    return document.querySelector("[data-chat]");
  }

  function chatActive() {
    return document.querySelector("#chat-thread [data-chat-answer][data-job-active]");
  }

  function atFoot(slack) {
    var page = document.documentElement;
    return window.innerHeight + window.scrollY >= page.scrollHeight - slack;
  }

  function toFoot() {
    window.scrollTo(0, document.documentElement.scrollHeight);
    chatLastY = window.scrollY;
  }

  function follow() {
    chatFollow = true;
    toFoot();
  }

  // The reader scrolled the page up (scrolling the side panel does not
  // count).
  function stopFollowing(event) {
    var inPanel = event && event.target instanceof Element && event.target.closest(".chat__panel");
    if (chatPage() && !inPanel) {
      chatFollow = false;
    }
  }

  function growInput(area) {
    area.style.height = "auto";
    area.style.height = Math.min(area.scrollHeight, window.innerHeight * 0.4) + "px";
  }

  function syncComposer() {
    var form = document.querySelector("[data-chat-composer]");
    if (!form) {
      return;
    }
    var busy = chatActive() !== null;
    form.classList.toggle("is-busy", busy);
    form.querySelector("[data-chat-send]").hidden = busy;
    form.querySelector("[data-chat-stop]").hidden = !busy;
  }

  // Details the reader opened or closed stay so when the card is replaced
  // (details[data-keep-open] with an id that stays).
  function keepOpen(scope) {
    scope.querySelectorAll("details[data-keep-open]").forEach(function (details) {
      if (Object.prototype.hasOwnProperty.call(keptOpen, details.id)) {
        details.open = keptOpen[details.id];
      }
    });
  }

  function chatBeforeUpdate() {
    if (!chatPage()) {
      return null;
    }
    var active = Array.prototype.map.call(
      document.querySelectorAll("#chat-thread [data-chat-answer][data-job-active]"),
      function (card) {
        return card.id;
      }
    );
    return { active: active };
  }

  function chatAfterUpdate(before) {
    if (!before) {
      return;
    }
    keepOpen(document);
    syncComposer();
    if (chatFollow) {
      toFoot();
    }
    var ended = before.active.some(function (id) {
      var card = document.getElementById(id);
      return card && !card.hasAttribute("data-job-active");
    });
    if (ended) {
      followUntil = Date.now() + FOLLOW_AFTER;
    }
    var page = chatPage();
    if (ended && page && window.htmx) {
      window.htmx.ajax("GET", page.getAttribute("data-chat-panel-url"), {
        target: "#chat-panel-body",
        swap: "outerHTML"
      });
    }
  }

  // An event stream that opens sends every card of the page; one the page
  // has in the same version stays, so its charts are not drawn again, nor
  // its player reset.
  document.addEventListener("htmx:oobBeforeSwap", function (event) {
    var target = event.detail.target;
    var incoming = event.detail.fragment ? event.detail.fragment.firstElementChild : null;
    if (
      target instanceof Element &&
      incoming &&
      target.hasAttribute("data-version") &&
      target.getAttribute("data-version") === incoming.getAttribute("data-version")
    ) {
      event.detail.shouldSwap = false;
    }
  });

  /* The reader scrolls up: the page stops following the reply. */
  window.addEventListener(
    "wheel",
    function (event) {
      if (event.deltaY < 0) {
        stopFollowing(event);
      }
    },
    { passive: true }
  );
  window.addEventListener(
    "touchstart",
    function (event) {
      touchY = event.touches.length ? event.touches[0].clientY : null;
    },
    { passive: true }
  );
  window.addEventListener(
    "touchmove",
    function (event) {
      if (touchY !== null && event.touches.length && event.touches[0].clientY > touchY + 4) {
        stopFollowing(event);
      }
    },
    { passive: true }
  );
  document.addEventListener("keydown", function (event) {
    var field = event.target instanceof Element && event.target.closest("input, textarea, select");
    if (!field && (event.key === "PageUp" || event.key === "ArrowUp" || event.key === "Home")) {
      stopFollowing(event);
    }
  });
  window.addEventListener(
    "scroll",
    function () {
      if (!chatPage()) {
        return;
      }
      var y = window.scrollY;
      if (atFoot(32)) {
        chatFollow = true;
      } else if (y < chatLastY - 1) {
        // The scroll bar dragged up: the page itself only scrolls down.
        chatFollow = false;
      }
      chatLastY = y;
    },
    { passive: true }
  );

  document.addEventListener("keydown", function (event) {
    var area = event.target instanceof Element ? event.target.closest("[data-chat-input]") : null;
    if (!area || event.key !== "Enter" || event.shiftKey || event.isComposing) {
      return;
    }
    event.preventDefault();
    var form = area.closest("form");
    if (area.value.trim() && !form.classList.contains("is-busy")) {
      form.requestSubmit();
    }
  });

  document.addEventListener("input", function (event) {
    if (event.target instanceof Element && event.target.matches("[data-chat-input]")) {
      growInput(event.target);
    }
  });

  document.addEventListener("htmx:afterRequest", function (event) {
    var element = event.detail.elt;
    if (!(element instanceof Element)) {
      return;
    }
    if (element.matches("[data-chat-attach]")) {
      // The same file can be chosen again after it was removed.
      element.querySelector("input[type='file']").value = "";
    } else if (element.matches("[data-chat-composer]") && event.detail.successful) {
      var area = element.querySelector("[data-chat-input]");
      area.value = "";
      growInput(area);
      syncComposer();
      follow();
    } else if (element.matches(".answer__again") && event.detail.successful) {
      follow();
    }
  });

  document.addEventListener("click", function (event) {
    var target = event.target instanceof Element ? event.target : null;
    if (!target) {
      return;
    }
    // A box the reader opened or closed stays so when its card is
    // replaced; the click opens or closes it after this.
    var summary = target.closest("details[data-keep-open] > summary");
    if (summary && summary.parentNode.id) {
      keptOpen[summary.parentNode.id] = !summary.parentNode.open;
      return;
    }
    if (target.closest("[data-chat-stop]")) {
      var card = chatActive();
      if (card && window.htmx) {
        window.htmx.ajax("POST", root + "/jobs/" + card.id.replace(/^job-/, "") + "/cancel", {
          target: card,
          swap: "outerHTML"
        });
      }
      return;
    }
    var toggle = target.closest("[data-chat-panel-toggle]");
    if (toggle) {
      var page = chatPage();
      var open = !page.classList.contains("is-panel-open");
      page.classList.toggle("is-panel-open", open);
      toggle.setAttribute("aria-expanded", String(open));
      return;
    }
    var copy = target.closest("[data-copy-code]");
    if (copy && navigator.clipboard) {
      var code = copy.parentNode.querySelector("pre");
      navigator.clipboard.writeText(code ? code.textContent : "").then(function () {
        copy.textContent = "Copied";
        window.setTimeout(function () {
          copy.textContent = "Copy";
        }, 1500);
      });
    }
  });

  /* Files dropped anywhere on the page are attached (and never opened
     by the browser in place of the page). */
  function draggingFiles(event) {
    return event.dataTransfer && Array.prototype.indexOf.call(event.dataTransfer.types, "Files") >= 0;
  }

  ["dragenter", "dragover"].forEach(function (type) {
    document.addEventListener(type, function (event) {
      var page = chatPage();
      if (page && draggingFiles(event)) {
        event.preventDefault();
        page.classList.add("is-dragover");
      }
    });
  });
  document.addEventListener("dragleave", function (event) {
    var page = chatPage();
    if (page && !(event.relatedTarget instanceof Node)) {
      page.classList.remove("is-dragover");
    }
  });
  document.addEventListener("drop", function (event) {
    var page = chatPage();
    if (!page || !draggingFiles(event)) {
      return;
    }
    event.preventDefault();
    page.classList.remove("is-dragover");
    if (event.dataTransfer.files.length) {
      var input = page.querySelector("[data-chat-file]");
      input.files = event.dataTransfer.files;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });

  /* Code in answers gets a Copy button. */
  function addCopyButtons(scope) {
    scope.querySelectorAll(".answer__text pre, .code-block pre").forEach(function (pre) {
      if (pre.parentNode.classList.contains("code")) {
        return;
      }
      var box = document.createElement("div");
      box.className = "code";
      pre.parentNode.insertBefore(box, pre);
      box.appendChild(pre);
      var button = document.createElement("button");
      button.type = "button";
      button.className = "text-button code__copy";
      button.setAttribute("data-copy-code", "");
      button.textContent = "Copy";
      box.appendChild(button);
    });
  }

  /* The data agents' interactive charts, drawn with plotly.js. */
  function loadPlotly(element) {
    if (window.Plotly) {
      return Promise.resolve(window.Plotly);
    }
    if (!plotlyLoading) {
      // plotly.js adds its style rules to the page unless this element is
      // there: the Content Security Policy would block them.
      if (!document.getElementById("plotly.js-style-global")) {
        var sheet = document.createElement("link");
        sheet.rel = "stylesheet";
        sheet.id = "plotly.js-style-global";
        sheet.className = "no-inline-styles";
        sheet.href = element.getAttribute("data-plotly-css");
        document.head.appendChild(sheet);
      }
      plotlyLoading = new Promise(function (resolve, reject) {
        var script = document.createElement("script");
        script.src = element.getAttribute("data-plotly");
        script.addEventListener("load", function () {
          resolve(window.Plotly);
        });
        script.addEventListener("error", function () {
          plotlyLoading = null;
          reject(new Error("plotly.js did not load"));
        });
        document.head.appendChild(script);
      });
    }
    return plotlyLoading;
  }

  function chartColours() {
    var style = window.getComputedStyle(document.body);
    var read = function (name) {
      return style.getPropertyValue(name).trim();
    };
    return {
      ink: read("--ink"),
      text: read("--ink-2"),
      rule: read("--rule"),
      strong: read("--rule-strong"),
      font: read("--font-sans")
    };
  }

  // The page's colours and type for a chart's layout, as relayout updates.
  function chartTheme(layout) {
    var colours = chartColours();
    var update = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      "font.family": colours.font,
      "font.color": colours.text,
      "title.font.color": colours.ink,
      "legend.bgcolor": "rgba(0,0,0,0)"
    };
    Object.keys(layout || {})
      .concat(["xaxis", "yaxis"])
      .forEach(function (key) {
        if (/^[xy]axis\d*$/.test(key)) {
          update[key + ".gridcolor"] = colours.rule;
          update[key + ".linecolor"] = colours.strong;
          update[key + ".zerolinecolor"] = colours.strong;
        }
      });
    return update;
  }

  function drawChart(element) {
    if (element.hasAttribute("data-drawn")) {
      return;
    }
    element.setAttribute("data-drawn", "");
    var figure = window.fetch(element.getAttribute("data-chart"), { credentials: "same-origin" }).then(
      function (response) {
        if (!response.ok) {
          throw new Error("status " + response.status);
        }
        return response.json();
      }
    );
    Promise.all([loadPlotly(element), figure])
      .then(function (loaded) {
        var Plotly = loaded[0];
        var layout = loaded[1].layout || {};
        element.textContent = "";
        return Plotly.newPlot(element, loaded[1].data || [], layout, {
          responsive: true,
          displaylogo: false
        }).then(function () {
          return Plotly.relayout(element, chartTheme(layout));
        });
      })
      .catch(function () {
        element.textContent = "The chart could not be drawn. It is in the downloads.";
      });
  }

  document.addEventListener("textlab:theme", function () {
    if (!window.Plotly) {
      return;
    }
    // The theme's colours apply once the style sheet has switched.
    window.requestAnimationFrame(function () {
      document.querySelectorAll("[data-chart][data-drawn].js-plotly-plot").forEach(function (element) {
        window.Plotly.relayout(element, chartTheme(element.layout));
      });
    });
  });
  if (window.matchMedia) {
    var darkScheme = window.matchMedia("(prefers-color-scheme: dark)");
    if (darkScheme.addEventListener) {
      darkScheme.addEventListener("change", function () {
        document.dispatchEvent(new CustomEvent("textlab:theme"));
      });
    }
  }

  // The data agents' results in new content: charts, code, open boxes.
  function initialiseResults(scope) {
    scope.querySelectorAll("[data-chart]").forEach(drawChart);
    addCopyButtons(scope);
    keepOpen(scope);
  }

  // Whatever changes the size of the conversation while a reply is
  // written, or of the message box while the reader types at the foot,
  // the foot stays in view.
  function watchChat() {
    var thread = document.getElementById("chat-thread");
    var composer = document.querySelector("[data-chat-composer]");
    if (!thread || !composer || !window.ResizeObserver) {
      return;
    }
    new ResizeObserver(function () {
      if (chatFollow && (chatActive() || Date.now() < followUntil)) {
        toFoot();
      }
    }).observe(thread);
    new ResizeObserver(function () {
      if (chatFollow) {
        toFoot();
      }
    }).observe(composer);
  }

  /* ---- Start ---------------------------------------------------------- */

  function initialise(scope) {
    scope.querySelectorAll("[data-reveal]").forEach(applyReveal);
    scope.querySelectorAll("[data-switch]").forEach(applySwitch);
    scope.querySelectorAll("input[type='range']").forEach(showRange);
    scope.querySelectorAll("select[data-readout], input[type='radio'][data-readout]").forEach(showReadout);
    initialiseResults(scope);
  }

  applyTheme(currentTheme());
  initialise(document);
  initialiseTranslator(document);
  if (chatPage()) {
    syncComposer();
    watchChat();
    follow();
  }
  if (window.htmx) {
    window.htmx.onLoad(function (element) {
      if (element instanceof Element && element !== document.body) {
        initialise(element);
      }
    });
  }
  syncStream();
})();
