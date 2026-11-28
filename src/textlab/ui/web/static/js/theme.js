/*
 * Applies the colour theme the user picked before the page is drawn, so
 * a dark page does not flash light first. Loaded in <head> without
 * "defer" for that reason; app.js has the switch itself.
 */
(function () {
  "use strict";
  try {
    var theme = window.localStorage.getItem("textlab-theme");
    if (theme === "light" || theme === "dark") {
      document.documentElement.setAttribute("data-theme", theme);
    }
  } catch (error) {
    // Storage can be blocked (private windows): follow the system then.
  }
})();
