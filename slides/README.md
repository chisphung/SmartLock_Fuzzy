# SmartLock Fuzzy Slides

This folder contains two slide formats:

- `index.html`: browser-viewable slide deck. Open it directly and use arrow keys to navigate.
- `smartlock_fuzzy_slides.tex`: Beamer source for PDF export.

Beamer source:

```bash
slides/smartlock_fuzzy_slides.tex
```

Compile the Beamer deck with XeLaTeX:

```bash
cd slides
xelatex smartlock_fuzzy_slides.tex
```

The deck reuses report images from `../document/`, including:

- `uit_logo.png`
- `haar_cascade.png`
- `lbph.png`
- `fuzzy_full.png`
- `schematic_connected.png`
- `dashboard_nextjs.png`
- `benchmark_ui.png`
