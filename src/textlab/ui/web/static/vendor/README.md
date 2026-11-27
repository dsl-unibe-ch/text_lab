# Vendored files

Everything the browser loads comes from this server: no CDN, no build step.
Each file is a pinned release from the npm registry, checked against the
registry's checksum before it was copied here, unchanged.

| File | Package and version | Licence | Checksum of the package |
|---|---|---|---|
| `htmx-2.0.11.min.js` | [`htmx.org`](https://www.npmjs.com/package/htmx.org) 2.0.11, `dist/htmx.min.js` | Zero-Clause BSD (`LICENSE-htmx`) | `sha512-Thx/WtpeOQqSrqBCw/A1cwGJGg4UrVa3+sW0GmrM3p4gJgO89ecH4qtbnyzDDWFvBTqjnIMCgELTNt636dtamA==`; the file's SHA-256 is `d6fdc75f204e6bdefa99b69bf1e6d4ac69b8a364f77929f45c13476b4000f717` |
| `lucide/*.svg` | [`lucide-static`](https://www.npmjs.com/package/lucide-static) 1.71.0, `icons/` | ISC (`lucide/LICENSE`) | `sha512-yUXCvkMCNlqZI72Oco8+FjFwTcPI0Xz5qPZuqYXKP/8BU928hKT0Q4/m+R0e1I44kqEfCz1pd2J7iQ8jTZQEuA==` |
| `ibm-plex-sans-5.3.0/*.woff2` | [`@fontsource-variable/ibm-plex-sans`](https://www.npmjs.com/package/@fontsource-variable/ibm-plex-sans) 5.3.0, `files/ibm-plex-sans-latin{,-ext}-standard-normal.woff2` (weights 100 to 700, widths 75% to 100%) | SIL Open Font License 1.1 (`ibm-plex-sans-5.3.0/LICENSE`) | `sha512-agG8tXFEo0hD9+J7npa4vbbWult52eMLVaQ6WQRlhs/iCAojrMAoejru85W9HTVXHfyUj96KM7gp/KGAS87XaQ==` |
| `plotly-3.7.0/plotly-strict.min.js` | [`plotly.js-strict-dist-min`](https://www.npmjs.com/package/plotly.js-strict-dist-min) 3.7.0, `plotly-strict.min.js` | MIT (`plotly-3.7.0/LICENSE`) | `sha512-icoHjKd4XUDxQyijcHgj1v2iujC+QF4s/HCj2iHn5SY52/wb2tPsHP5wbtq6Vv/qW0qrqWmCjpxHRR0tfptRmw==`; the file's SHA-256 is `ca8715e7e348e1d56fb5d31575c7850b5cdd277f601bf047c0fefc4172e2957b` |
| `ibm-plex-mono-5.3.0/*.woff2` | [`@fontsource/ibm-plex-mono`](https://www.npmjs.com/package/@fontsource/ibm-plex-mono) 5.3.0, `files/ibm-plex-mono-latin{,-ext}-{400,500}-normal.woff2` | SIL Open Font License 1.1 (`ibm-plex-mono-5.3.0/LICENSE`) | `sha512-eTgnZjZEGk1QtD3ZstF+Vclo2HLAni8YMy34/DxllwZvyz1lR/1RF/xTiAquOBO7MvqBx8D2Ig2WCPMVfdZu7Q==` |

htmx 2.0.11 is the release npm tags `latest`; htmx 4 is tagged `next`.
Lucide releases often; 1.71.0 was two weeks old when it was taken. The
fonts are the Latin and Latin Extended files of their packages;
`css/fonts.css` declares them, and browsers load the Latin Extended file
only for text that needs it.

Vendored files carry their release in their file or folder name, so the
server lets browsers cache them for good (`app.StaticAssets`). A new
release goes into a new folder.

## Icons

Only the icons the templates use are copied: the design uses few icons
(arrows, download, play and pause, close, menu, the select's chevron, and
for Chat: attach, send, stop, new chat, answer again). The templates show them with
`icon("name")` (see `render.py`), which inlines the SVG so it takes the
colour of the text around it. A test fails if a template names an icon
that is not here, or if one here is not used.

To add an icon, take it from the same release, so the set stays one
version:

```bash
version=1.71.0
want="sha512-yUXCvkMCNlqZI72Oco8+FjFwTcPI0Xz5qPZuqYXKP/8BU928hKT0Q4/m+R0e1I44kqEfCz1pd2J7iQ8jTZQEuA=="
work="$(mktemp -d)"
curl -sSo "$work/lucide.tgz" "https://registry.npmjs.org/lucide-static/-/lucide-static-$version.tgz"
got="sha512-$(openssl dgst -sha512 -binary "$work/lucide.tgz" | openssl base64 -A)"
[[ "$got" == "$want" ]] || { echo "checksum mismatch"; exit 1; }
tar -xzf "$work/lucide.tgz" -C "$work" package/icons/NAME.svg
cp "$work/package/icons/NAME.svg" lucide/
```

Browse the icons at <https://lucide.dev/icons/>.

## plotly.js

Chat draws the data agents' interactive charts with plotly.js; Visualize
Data will too. `app.js` loads it only on a page with such a chart.

- **The version** is the one the image's plotly.py writes its figures for
  (plotly 6.9.0 bundles plotly.js 3.7.0; see
  `plotly.offline.get_plotlyjs_version()`), not npm's latest. Change it
  with plotly.py.
- **The strict bundle**, so the Content Security Policy needs no
  `unsafe-eval`: its WebGL charts (such as a scatter matrix) are built
  without `new Function`. The few `Function` calls left are fallbacks
  modern browsers never reach.
- **Its style rules** (`plotly.css`): plotly.js would add them in a
  `<style>` element, which the policy blocks. It skips that when the page
  holds an element with the id `plotly.js-style-global` and the class
  `no-inline-styles`, so `app.js` links `plotly.css` under that id first.
  The file is read from the bundle, so it matches its version:

  ```bash
  python3 scripts/plotly_css.py \
      src/textlab/ui/web/static/vendor/plotly-3.7.0/plotly-strict.min.js \
      > src/textlab/ui/web/static/vendor/plotly-3.7.0/plotly.css
  ```

  A test checks that the file is what the script writes.

Taking the bundle, as for the icons:

```bash
version=3.7.0
want="sha512-icoHjKd4XUDxQyijcHgj1v2iujC+QF4s/HCj2iHn5SY52/wb2tPsHP5wbtq6Vv/qW0qrqWmCjpxHRR0tfptRmw=="
work="$(mktemp -d)"
curl -sSo "$work/plotly.tgz" "https://registry.npmjs.org/plotly.js-strict-dist-min/-/plotly.js-strict-dist-min-$version.tgz"
got="sha512-$(openssl dgst -sha512 -binary "$work/plotly.tgz" | openssl base64 -A)"
[[ "$got" == "$want" ]] || { echo "checksum mismatch"; exit 1; }
tar -xzf "$work/plotly.tgz" -C "$work"
mkdir -p "plotly-$version"
cp "$work/package/plotly-strict.min.js" "$work/package/LICENSE" "plotly-$version/"
```

## Updating htmx

Take a release npm tags `latest` that is at least two weeks old, check the
tarball against the registry's `dist.integrity`, copy `dist/htmx.min.js`
here under its version, and change the `<script>` in
`templates/base.html` and this table.
