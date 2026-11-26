# Vendored files

Everything the browser loads comes from this server: no CDN, no build step.
Each file is a pinned release from the npm registry, checked against the
registry's checksum before it was copied here, unchanged.

| File | Package and version | Licence | Checksum of the package |
|---|---|---|---|
| `htmx-2.0.11.min.js` | [`htmx.org`](https://www.npmjs.com/package/htmx.org) 2.0.11, `dist/htmx.min.js` | Zero-Clause BSD (`LICENSE-htmx`) | `sha512-Thx/WtpeOQqSrqBCw/A1cwGJGg4UrVa3+sW0GmrM3p4gJgO89ecH4qtbnyzDDWFvBTqjnIMCgELTNt636dtamA==`; the file's SHA-256 is `d6fdc75f204e6bdefa99b69bf1e6d4ac69b8a364f77929f45c13476b4000f717` |
| `lucide/*.svg` | [`lucide-static`](https://www.npmjs.com/package/lucide-static) 1.71.0, `icons/` | ISC (`lucide/LICENSE`) | `sha512-yUXCvkMCNlqZI72Oco8+FjFwTcPI0Xz5qPZuqYXKP/8BU928hKT0Q4/m+R0e1I44kqEfCz1pd2J7iQ8jTZQEuA==` |
| `ibm-plex-sans-5.3.0/*.woff2` | [`@fontsource-variable/ibm-plex-sans`](https://www.npmjs.com/package/@fontsource-variable/ibm-plex-sans) 5.3.0, `files/ibm-plex-sans-latin{,-ext}-standard-normal.woff2` (weights 100 to 700, widths 75% to 100%) | SIL Open Font License 1.1 (`ibm-plex-sans-5.3.0/LICENSE`) | `sha512-agG8tXFEo0hD9+J7npa4vbbWult52eMLVaQ6WQRlhs/iCAojrMAoejru85W9HTVXHfyUj96KM7gp/KGAS87XaQ==` |
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
(arrows, download, play and pause, close, menu, the select's chevron). The templates show them with
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

## Updating htmx

Take a release npm tags `latest` that is at least two weeks old, check the
tarball against the registry's `dist.integrity`, copy `dist/htmx.min.js`
here under its version, and change the `<script>` in
`templates/base.html` and this table.
