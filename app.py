"""
Dash Imaging App — in-memory store + HTTP ingestion (process-safe)

This version solves cross-process updates by exposing /api/add_shot.
Producers POST shots to the app; the UI poller notices the new version
and refreshes selectors/plots. For embedded (same-process) use, you
can still import and call add_shot(...) directly.

Payload format for /api/add_shot (JSON):
{
  "meta": { ... },
  "frames_npz_b64": "<base64 of np.savez_compressed(**frames)>"
}
Optionally (for tiny arrays only), you may send:
{
  "meta": { ... },
  "frames": { "name": [[...], ...], ... }  # will be coerced to float32
}

Optional security: set INGEST_TOKEN (env var or module constant) and
include header 'X-Ingest-Token: <token>' in client requests.

Notes
- Keep images 2D float arrays. Downsample/compress on the producer if needed.
- HISTORY_SIZE limits memory. Eviction drops oldest meta row as well.
"""

from __future__ import annotations
import base64
import io
import os
import time
import threading
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from flask import request, jsonify

# =========================
# ------ Configs ----------
# =========================

HISTORY_SIZE = int(os.getenv("HISTORY_SIZE", "50"))
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")  # empty = no auth

COLOR_SCALES = [
    "Gray", "Viridis", "Cividis", "Plasma", "Magma",
    "Turbo", "Inferno", "Blues", "Reds"
]

# =========================
# ---- Data Store ----------
# =========================

class DataStore:
    """Thread-safe rolling store for shots and pooled meta."""
    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self._lock = threading.Lock()
        self._shots: List[Dict[str, Any]] = []
        self._meta = pd.DataFrame()
        self._version = 0
        self._next_shot_id = 1

    # --- public API ---
    def add_shot(self, *, frames: Dict[str, np.ndarray], meta: Dict[str, Any], shot_name: str | None = None) -> int:
        with self._lock:
            shot_id = self._next_shot_id
            self._next_shot_id += 1
            name = shot_name if (shot_name is not None and str(shot_name) != "") else f"Shot {shot_id}"
            shot = {
                "frames": frames,
                "meta": dict(meta),
                "shot_id": shot_id,
                "shot_name": name,  # <-- store the human-friendly name
                "t": time.time(),
            }
            self._shots.append(shot)

            # FIFO eviction (unchanged)
            if len(self._shots) > self.history_size:
                drop_id = self._shots[0]["shot_id"]
                self._shots = self._shots[-self.history_size:]
                if not self._meta.empty:
                    self._meta = self._meta[self._meta["shot_id"] != drop_id]

            # append meta (also keep shot_name for grouping/filtering if desired)
            mr = dict(meta)
            mr["shot_id"] = shot_id
            mr["shot_name"] = name
            self._meta = pd.concat([self._meta, pd.DataFrame([mr])], ignore_index=True)

            self._version += 1
            return shot_id

    def reset(self) -> None:
        with self._lock:
            self._shots.clear()
            self._meta = pd.DataFrame()
            self._version += 1

    def snapshot(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame, int]:
        """Shallow copy (frames are read-only)."""
        with self._lock:
            return list(self._shots), self._meta.copy(), self._version


# Single global store (works for embedded same-process calls)
store = DataStore(history_size=HISTORY_SIZE)

# Convenience functions for embedded use
# REPLACE the embedded add_shot(...) wrapper so it passes shot_name through
def add_shot(data: Dict[str, Any]) -> int:
    frames = data.get("frames", {})
    meta = data.get("meta", {})
    shot_name = data.get("shot_name")  # <-- new
    if not isinstance(frames, dict) or not isinstance(meta, dict):
        raise TypeError("Expected dicts for 'frames' and 'meta'.")
    for k, v in frames.items():
        if not (isinstance(v, np.ndarray) and v.ndim == 2):
            raise ValueError(f"Frame '{k}' must be 2D numpy array.")
        if v.dtype not in (np.float32, np.float64):
            frames[k] = v.astype(np.float32, copy=False)
    return store.add_shot(frames=frames, meta=meta, shot_name=shot_name)  # <-- pass along


def reset_pool() -> None:
    store.reset()

# =========================
# -- Serialization utils --
# =========================

def frames_to_b64_npz(frames: Dict[str, np.ndarray]) -> str:
    """Serialize frames dict to base64(np.savez_compressed)."""
    for k, v in frames.items():
        if not isinstance(v, np.ndarray) or v.ndim != 2:
            raise ValueError(f"Frame '{k}' must be 2D numpy array.")
    buf = io.BytesIO()
    np.savez_compressed(buf, **frames)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def b64_npz_to_frames(b64: str) -> Dict[str, np.ndarray]:
    """Deserialize frames dict from base64(np.savez_compressed)."""
    raw = base64.b64decode(b64.encode("ascii"))
    with np.load(io.BytesIO(raw)) as npz:
        return {k: np.array(npz[k]) for k in npz.files}

# =========================
# ------- Dash UI ---------
# =========================

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Cold Atom Imaging Dashboard",
)
server = app.server  # Flask app

# --- Controls (accordion for future groups) ---
controls = dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                html.Div([
                    # --- Selection ---
                    html.H6("Selection", className="mt-1 mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Shot", className="mb-0 pt-2"), width=3),
                        dbc.Col(dcc.Dropdown(id="shot-select", options=[], value=None, clearable=False), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Frame", className="mb-0 pt-2"), width=3),
                        dbc.Col(dcc.Dropdown(id="frame-select", options=[], value=None, clearable=False), width=9),
                    ], className="mb-3"),

                    # --- Image Display ---
                    html.H6("Image Display", className="mt-1 mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Colormap", className="mb-0 pt-2"), width=3),
                        dbc.Col(dcc.Dropdown(
                            id="cmap",
                            options=[{"label": c, "value": c} for c in COLOR_SCALES],
                            value="Gray",
                            clearable=False,
                        ), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Autoscale", className="mb-0 pt-2"), width=3),
                        dbc.Col(dbc.Checklist(
                            options=[{"label": "Auto", "value": "auto"}],
                            value=["auto"],
                            id="autoscale",
                            inline=True,
                            className="pt-1",
                        ), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("vmin", className="mb-0 pt-2"), width=3),
                        dbc.Col(dbc.Input(id="vmin", type="number", debounce=True, placeholder="auto"), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("vmax", className="mb-0 pt-2"), width=3),
                        dbc.Col(dbc.Input(id="vmax", type="number", debounce=True, placeholder="auto"), width=9),
                    ], className="mb-3"),

                    # --- 1D Plot ---
                    html.H6("1D Plot", className="mt-1 mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("X", className="mb-0 pt-2"), width=3),
                        dbc.Col(dcc.Dropdown(id="x-col", options=[], value=None, clearable=True), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Y", className="mb-0 pt-2"), width=3),
                        dbc.Col(dcc.Dropdown(id="y-col", options=[], value=None, clearable=True), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Group by", className="mb-0 pt-2"), width=3),
                        dbc.Col(dcc.Dropdown(id="group-col", options=[], value=None, clearable=True), width=9),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Label("Scales", className="mb-0 pt-2"), width=3),
                        dbc.Col(
                            dbc.Checklist(
                                options=[{"label": "log X", "value": "logx"}, {"label": "log Y", "value": "logy"}],
                                value=[],
                                id="log-checks",
                                inline=True,
                                className="pt-1",
                            ),
                            width=9,
                        ),
                    ], className="mb-2"),
                    # keep IDs expected by callbacks:
                    dcc.Store(id="logx", data=[]),
                    dcc.Store(id="logy", data=[]),
                    dcc.Store(id="avg", data=[]),

                    # dbc.Row([
                    #     dbc.Col(dbc.Label("Average", className="mb-0 pt-2"), width=3),
                    #     dbc.Col(dbc.Checklist(
                    #         options=[{"label": "Average", "value": "avg"}],
                    #         value=[],
                    #         id="avg",
                    #         inline=True,
                    #         className="pt-1",
                    #     ), width=9),
                    # ], className="mb-3"),

                    # --- Actions ---
                    dbc.Row([
                        dbc.Col(dbc.Button("Clear Data Pool", id="clear-btn", color="danger", className="w-100"), width=12),
                    ]),
                ])
            ],
            title="Primary Controls",
            item_id="controls-primary",
        ),
    ],
    start_collapsed=False,
    always_open=True,
    id="controls-accordion",
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id="data-version", data=0),
        dcc.Interval(id="poller", interval=100, n_intervals=0),  # UI refresh cadence (ms)
        html.H3("Cold Atom Imaging Dashboard", className="my-2"),
        dbc.Row([
            dbc.Col(controls, width=3),
            dbc.Col([
                dcc.Graph(id="image-fig", style={"height": "48vh"}),
                dcc.Graph(id="plot-fig", style={"height": "42vh"}),
            ], width=9),
        ], align="start"),
    ],
)

# =========================
# ------- Endpoints -------
# =========================

@server.post("/api/add_shot")
def api_add_shot():
    if INGEST_TOKEN and request.headers.get("X-Ingest-Token", "") != INGEST_TOKEN:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"ok": False, "error": "invalid JSON"}), 400

    meta = data.get("meta", {}) or {}
    shot_name = data.get("shot_name")  # <-- new
    frames: Dict[str, np.ndarray] = {}

    if "frames_npz_b64" in data and data["frames_npz_b64"]:
        try:
            frames = b64_npz_to_frames(data["frames_npz_b64"])
        except Exception as e:
            return jsonify({"ok": False, "error": f"bad frames_npz_b64: {e}"}), 400
    elif "frames" in data and isinstance(data["frames"], dict):
        for k, v in data["frames"].items():
            arr = np.array(v, dtype=np.float32)
            if arr.ndim != 2:
                return jsonify({"ok": False, "error": f"frame {k} must be 2D"}), 400
            frames[k] = arr
    # else: meta-only shot allowed

    try:
        shot_id = store.add_shot(frames=frames, meta=meta, shot_name=shot_name)  # <-- pass along
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    _, _, ver = store.snapshot()
    return jsonify({"ok": True, "shot_id": shot_id, "version": ver})

@server.post("/api/reset_pool")
def api_reset_pool():
    if INGEST_TOKEN:
        if request.headers.get("X-Ingest-Token", "") != INGEST_TOKEN:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
    store.reset()
    _, _, ver = store.snapshot()
    return jsonify({"ok": True, "version": ver})

# =========================
# ------- Callbacks -------
# =========================

@app.callback(
    Output("data-version", "data"),
    Output("shot-select", "options"),
    Output("shot-select", "value"),
    Output("frame-select", "options"),
    Output("frame-select", "value"),
    Output("x-col", "options"),
    Output("y-col", "options"),
    Output("group-col", "options"),
    Input("poller", "n_intervals"),
    State("data-version", "data"),
    prevent_initial_call=False,
)
def refresh_options(_tick, last_seen_version):
    shots, meta, version = store.snapshot()
    if version == last_seen_version and _tick != 0:
        raise PreventUpdate

    # Shot options (newest first), using shot_name as both label and value
    shot_opts = []
    if shots:
        for s in reversed(shots):
            name = s.get("shot_name") or f"Shot {s['shot_id']}"
            shot_opts.append({"label": name, "value": name})

        # ALWAYS jump to the newest shot on data change
        shot_value = shot_opts[0]["value"]
        # find selected shot by shot_name (fallback to formatted id name)
        sel = next(
            (s for s in shots if s.get("shot_name") == shot_value or f"Shot {s['shot_id']}" == shot_value),
            None
        )
        frame_names = list(sel["frames"].keys()) if (sel and isinstance(sel["frames"], dict)) else []
    else:
        shot_value = None
        frame_names = []

    frame_opts = [{"label": f, "value": f} for f in frame_names]
    frame_value = frame_names[0] if frame_names else None

    # Meta columns (exclude shot_id)
    meta_cols = [c for c in meta.columns if c != "shot_id"]
    meta_opts = [{"label": c, "value": c} for c in meta_cols]

    return (
        version,
        shot_opts, shot_value,
        frame_opts, frame_value,
        meta_opts, meta_opts, meta_opts,
    )

@app.callback(
    Output("image-fig", "figure"),
    Input("shot-select", "value"),
    Input("frame-select", "value"),
    Input("cmap", "value"),
    Input("autoscale", "value"),
    Input("vmin", "value"),
    Input("vmax", "value"),
    Input("data-version", "data"),
)
def update_image(shot_value, frame_value, cmap, autoscale_vals, vmin, vmax, _ver):
    shots, _meta, _version = store.snapshot()

    if not shots or shot_value is None or frame_value is None:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No image available",
                          margin=dict(l=10, r=10, t=30, b=10))
        return fig

    # find selected shot by shot_name (fallback to formatted id name)
    sel = next(
        (s for s in shots if s.get("shot_name") == shot_value or f"Shot {s['shot_id']}" == shot_value),
        None
    )
    if not sel or frame_value not in sel["frames"]:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Frame not found")
        return fig

    img = sel["frames"][frame_value]
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Invalid image")
        return fig

    use_auto = "auto" in (autoscale_vals or [])
    zmin = None if use_auto or vmin is None else float(vmin)
    zmax = None if use_auto or vmax is None else float(vmax)

    fig = px.imshow(
        img,
        color_continuous_scale=cmap or "Gray",
        aspect="equal",
        zmin=zmin,
        zmax=zmax,
        origin="upper",
    )
    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(title="ADU"),
        title=f"{shot_value} — Frame: {frame_value}",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

@app.callback(
    Output("plot-fig", "figure"),
    Input("x-col", "value"),
    Input("y-col", "value"),
    Input("group-col", "value"),
    Input("logx", "value"),
    Input("logy", "value"),
    Input("avg", "value"),
    Input("data-version", "data"),
)
def update_1d_plot(x_col, y_col, group_col, logx_vals, logy_vals, avg_vals, _ver):
    shots, meta, _ = store.snapshot()

    if meta.empty or x_col is None or y_col is None:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Select X and Y columns",
                          margin=dict(l=10, r=10, t=30, b=10))
        return fig

    df = meta.copy()

    # numeric coercion for X/Y
    def _to_num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    df["_X_"] = df[x_col].apply(_to_num)
    df["_Y_"] = df[y_col].apply(_to_num)
    df = df.dropna(subset=["_X_", "_Y_"])

    average = "avg" in (avg_vals or [])
    if average:
        group_fields = ["_X_"] + ([group_col] if group_col else [])
        g = df.groupby(group_fields, dropna=False, as_index=False)["_Y_"].mean()
        if group_col:
            fig = px.line(g.sort_values(["_X_", group_col]), x="_X_", y="_Y_", color=group_col,
                          markers=False, labels={"_X_": x_col, "_Y_": y_col})
        else:
            fig = px.line(g.sort_values("_X_"), x="_X_", y="_Y_",
                          markers=False, labels={"_X_": x_col, "_Y_": y_col})
        title = f"{y_col} vs {x_col} (avg)"
    else:
        if group_col:
            fig = px.scatter(df, x="_X_", y="_Y_", color=group_col,
                             labels={"_X_": x_col, "_Y_": y_col})
        else:
            fig = px.scatter(df, x="_X_", y="_Y_",
                             labels={"_X_": x_col, "_Y_": y_col})
        title = f"{y_col} vs {x_col}"

    fig.update_layout(template="plotly_white", title=title,
                      margin=dict(l=10, r=10, t=30, b=10))
    if "logx" in (logx_vals or []):
        fig.update_xaxes(type="log")
    if "logy" in (logy_vals or []):
        fig.update_yaxes(type="log")
    return fig

@app.callback(
    Output("clear-btn", "n_clicks"),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def on_clear(n_clicks):
    if n_clicks:
        store.reset()
        return 0
    return n_clicks

# =========================
# ---- Smoke seed + run ---
# =========================

if __name__ == "__main__":
    # rng = np.random.default_rng(42)
    # for i in range(3):
    #     frames = {
    #         "raw": rng.normal(1000 + 50*i, 50, size=(128, 128)).astype(np.float32),
    #         "proc": rng.normal(200 + 10*i, 20, size=(128, 128)).astype(np.float32),
    #     }
    #     meta = {"B_field_G": 10 + i, "lattice_depth_Er": 4.5 + 0.5*i, "note": f"test{i}", "run_tag": "demo"}
    #     add_shot({"frames": frames, "meta": meta})

    # For remote producers, expose on 0.0.0.0 and consider TLS/reverse proxy.
    app.run(debug=True, host="127.0.0.1", port=8050)
