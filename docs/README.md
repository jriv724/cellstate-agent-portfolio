# CellState Agent scientific site — v8

Preview locally:

```bash
cd cellstate-agent-science-demo-v8
python3 -m http.server 8000
```

Open:

`http://localhost:8000`

## Embedded recordings

- `assets/01_capabilities_and_options.mp4`
- `assets/02_live_analysis_workflow.mp4`
- `assets/03_results_and_report.mp4`

All videos use `preload="metadata"` so the page does not download the full recordings
until the reviewer chooses to play them.

## Scientific report link

The report button points to:

`assets/cellstate_agent_scientific_report.pdf`

Copy the final public-safe report to that path before publishing GitHub Pages. Because
the link is relative, it will work both locally and from the repository's published site.

## File-size note

The supplied live-analysis recording is approximately 89 MB and about 21 minutes,
44 seconds long. It is embedded as provided. For the public repository, the planned
short accelerated export will load faster and better match the intended viewing time.
