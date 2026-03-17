from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from WebServer.setting.setting_config import (
    DEFAULT_SETTINGS,
    FONT_SIZE_SMALL, FONT_SIZE_MEDIUM, FONT_SIZE_LARGE,
    IMAGE_SIZE_SMALL, IMAGE_SIZE_MEDIUM, IMAGE_SIZE_LARGE,
    DURATION_SHORT, DURATION_MEDIUM, DURATION_LONG,
    VOICE_MALE, VOICE_FEMALE,
)

router = APIRouter()

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"

ALLOWED_FONT = {FONT_SIZE_SMALL, FONT_SIZE_MEDIUM, FONT_SIZE_LARGE}
ALLOWED_IMAGE = {IMAGE_SIZE_SMALL, IMAGE_SIZE_MEDIUM, IMAGE_SIZE_LARGE}
ALLOWED_DURATION = {DURATION_SHORT, DURATION_MEDIUM, DURATION_LONG}
ALLOWED_VOICE = {VOICE_MALE, VOICE_FEMALE}
PANELS = {"cue_display", "cue_audio", "cue_selection"}

CUE_NAME = "name"
CUE_RELATIONSHIP = "relationship"
CUE_IMAGE = "image"
CUE_AUDIO = "audio"
ALLOWED_CUE_SELECTIONS = {CUE_NAME, CUE_RELATIONSHIP, CUE_IMAGE, CUE_AUDIO}
DEFAULT_CUE_SELECTIONS = [CUE_NAME, CUE_RELATIONSHIP, CUE_IMAGE, CUE_AUDIO]


def _default_settings() -> dict:
    base = dict(DEFAULT_SETTINGS)
    base["cue_selection"] = list(DEFAULT_CUE_SELECTIONS)
    return {
        "font_size": base.get("font_size", FONT_SIZE_MEDIUM),
        "image_size": base.get("image_size", IMAGE_SIZE_MEDIUM),
        "duration_time": base.get("duration_time", DURATION_MEDIUM),
        "voice_type": base.get("voice_type", VOICE_FEMALE),
        "cue_selection": list(base.get("cue_selection", DEFAULT_CUE_SELECTIONS)),
    }


def _normalize_settings(data: dict) -> dict:
    defaults = _default_settings()

    font_size = data.get("font_size", defaults["font_size"])
    image_size = data.get("image_size", defaults["image_size"])
    duration_time = data.get("duration_time", defaults["duration_time"])
    voice_type = data.get("voice_type", defaults["voice_type"])
    cue_selection = data.get("cue_selection", defaults["cue_selection"])

    if font_size not in ALLOWED_FONT:
        font_size = defaults["font_size"]
    if image_size not in ALLOWED_IMAGE:
        image_size = defaults["image_size"]
    if duration_time not in ALLOWED_DURATION:
        duration_time = defaults["duration_time"]
    if voice_type not in ALLOWED_VOICE:
        voice_type = defaults["voice_type"]

    if not isinstance(cue_selection, list):
        cue_selection = list(defaults["cue_selection"])
    else:
        cue_selection = [item for item in cue_selection if item in ALLOWED_CUE_SELECTIONS]
        if not cue_selection:
            cue_selection = list(defaults["cue_selection"])

    return {
        "font_size": font_size,
        "image_size": image_size,
        "duration_time": duration_time,
        "voice_type": voice_type,
        "cue_selection": cue_selection,
    }


def _save_settings(data: dict) -> None:
    normalized = _normalize_settings(data)
    SETTINGS_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def _load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        base = _default_settings()
        _save_settings(base)
        return base

    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        base = _default_settings()
        _save_settings(base)
        return base

    normalized = _normalize_settings(data)

    if normalized != data:
        _save_settings(normalized)

    return normalized


def _sel(panel: str, current: str) -> str:
    return "selected" if panel == current else ""


class SettingsUpdate(BaseModel):
    font_size: Optional[str] = None
    image_size: Optional[str] = None
    duration_time: Optional[str] = None
    voice_type: Optional[str] = None
    cue_selection: Optional[list[str]] = None


@router.post("/setting/api/settings")
def api_update_settings(payload: SettingsUpdate):
    current = _load_settings()
    update_data = payload.model_dump(exclude_unset=True)

    if "font_size" in update_data and update_data["font_size"] not in ALLOWED_FONT:
        return JSONResponse({"ok": False, "error": "Invalid font_size"}, status_code=400)

    if "image_size" in update_data and update_data["image_size"] not in ALLOWED_IMAGE:
        return JSONResponse({"ok": False, "error": "Invalid image_size"}, status_code=400)

    if "duration_time" in update_data and update_data["duration_time"] not in ALLOWED_DURATION:
        return JSONResponse({"ok": False, "error": "Invalid duration_time"}, status_code=400)

    if "voice_type" in update_data and update_data["voice_type"] not in ALLOWED_VOICE:
        return JSONResponse({"ok": False, "error": "Invalid voice_type"}, status_code=400)

    if "cue_selection" in update_data:
        if not isinstance(update_data["cue_selection"], list):
            return JSONResponse({"ok": False, "error": "Invalid cue_selection"}, status_code=400)

        cue_selection = [
            item for item in update_data["cue_selection"]
            if item in ALLOWED_CUE_SELECTIONS
        ]
        if not cue_selection:
            cue_selection = list(DEFAULT_CUE_SELECTIONS)
        update_data["cue_selection"] = cue_selection

    merged = {**current, **update_data}
    _save_settings(merged)
    return {"ok": True}


@router.get("/setting", response_class=HTMLResponse)
def setting_page(request: Request, panel: str = "cue_display"):
    if panel not in PANELS:
        panel = "cue_display"

    s = _load_settings()
    font_size = s["font_size"]
    image_size = s["image_size"]
    duration_time = s["duration_time"]
    voice_type = s["voice_type"]
    cue_selection = s["cue_selection"]

    current_settings_json = json.dumps(s)

    left_nav_html = f"""
        <a class="nav-link" href="/setting?panel=cue_display">
            <div class="nav-item {_sel("cue_display", panel)}">Cue Display</div>
        </a>
        <a class="nav-link" href="/setting?panel=cue_audio">
            <div class="nav-item {_sel("cue_audio", panel)}">Cue Audio</div>
        </a>
        <a class="nav-link" href="/setting?panel=cue_selection">
            <div class="nav-item {_sel("cue_selection", panel)}">Cue Selection</div>
        </a>
    """

    if panel == "cue_display":
        right_html = f"""
        <div class="row">
            <div class="label">Font Size</div>

            <div class="tri" data-kind="font">
                <input type="radio" id="font-small"  name="font_size" value="{FONT_SIZE_SMALL}"  {"checked" if font_size == FONT_SIZE_SMALL else ""}/>
                <input type="radio" id="font-medium" name="font_size" value="{FONT_SIZE_MEDIUM}" {"checked" if font_size == FONT_SIZE_MEDIUM else ""}/>
                <input type="radio" id="font-large"  name="font_size" value="{FONT_SIZE_LARGE}"  {"checked" if font_size == FONT_SIZE_LARGE else ""}/>

                <div class="track" data-track="font">
                    <div class="line"></div>
                    <div class="tick t0"></div>
                    <div class="tick t1"></div>
                    <div class="tick t2"></div>
                    <div class="dot"></div>
                </div>

                <div class="ticks">
                    <label for="font-small">Small</label>
                    <label for="font-medium">Medium</label>
                    <label for="font-large">Large</label>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="label">Image Size</div>

            <div class="tri" data-kind="image">
                <input type="radio" id="img-small"  name="image_size" value="{IMAGE_SIZE_SMALL}"  {"checked" if image_size == IMAGE_SIZE_SMALL else ""}/>
                <input type="radio" id="img-medium" name="image_size" value="{IMAGE_SIZE_MEDIUM}" {"checked" if image_size == IMAGE_SIZE_MEDIUM else ""}/>
                <input type="radio" id="img-large"  name="image_size" value="{IMAGE_SIZE_LARGE}"  {"checked" if image_size == IMAGE_SIZE_LARGE else ""}/>

                <div class="track" data-track="image">
                    <div class="line"></div>
                    <div class="tick t0"></div>
                    <div class="tick t1"></div>
                    <div class="tick t2"></div>
                    <div class="dot"></div>
                </div>

                <div class="ticks">
                    <label for="img-small">Small</label>
                    <label for="img-medium">Medium</label>
                    <label for="img-large">Large</label>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="label">Cue Duration</div>

            <div class="tri" data-kind="duration">
                <input type="radio" id="dur-short"  name="duration_time" value="{DURATION_SHORT}"  {"checked" if duration_time == DURATION_SHORT else ""}/>
                <input type="radio" id="dur-medium" name="duration_time" value="{DURATION_MEDIUM}" {"checked" if duration_time == DURATION_MEDIUM else ""}/>
                <input type="radio" id="dur-long"  name="duration_time" value="{DURATION_LONG}"  {"checked" if duration_time == DURATION_LONG else ""}/>

                <div class="track" data-track="duration">
                    <div class="line"></div>
                    <div class="tick t0"></div>
                    <div class="tick t1"></div>
                    <div class="tick t2"></div>
                    <div class="dot"></div>
                </div>

                <div class="ticks">
                    <label for="dur-short">Short</label>
                    <label for="dur-medium">Medium</label>
                    <label for="dur-long">Long</label>
                </div>
            </div>
        </div>
        """
    elif panel == "cue_audio":
        right_html = f"""
        <div class="row">
            <div class="label">Voice Type</div>

            <div class="toggle-group">
                <input type="radio" id="voice-female" name="voice_type" value="{VOICE_FEMALE}" {"checked" if voice_type == VOICE_FEMALE else ""}/>
                <label for="voice-female" class="toggle-option">Female</label>

                <input type="radio" id="voice-male" name="voice_type" value="{VOICE_MALE}" {"checked" if voice_type == VOICE_MALE else ""}/>
                <label for="voice-male" class="toggle-option">Male</label>
            </div>
        </div>
        """
    else:
        right_html = f"""
        <div class="cue-selection-panel">
            <div class="section-title">Cue Selection</div>
            <div class="section-description">Select the cue(s) you want to display.</div>

            <div class="checkbox-group">
                <input type="checkbox" id="cue-name" name="cue_selection" value="{CUE_NAME}" {"checked" if CUE_NAME in cue_selection else ""}/>
                <label for="cue-name" class="checkbox-option">Name</label>

                <input type="checkbox" id="cue-relationship" name="cue_selection" value="{CUE_RELATIONSHIP}" {"checked" if CUE_RELATIONSHIP in cue_selection else ""}/>
                <label for="cue-relationship" class="checkbox-option">Relationship</label>

                <input type="checkbox" id="cue-image" name="cue_selection" value="{CUE_IMAGE}" {"checked" if CUE_IMAGE in cue_selection else ""}/>
                <label for="cue-image" class="checkbox-option">Image</label>

                <input type="checkbox" id="cue-audio" name="cue_selection" value="{CUE_AUDIO}" {"checked" if CUE_AUDIO in cue_selection else ""}/>
                <label for="cue-audio" class="checkbox-option">Audio</label>
            </div>
        </div>
        """

    html = f"""
    <html>
    <head>
        <title>Setting</title>
        <style>
            :root {{
                --navy: #0b2a5b;
                --border: #1f4f7a;
                --selected: #aeb7c7;
                --white: #ffffff;
            }}

            body {{
                margin: 0;
                font-family: Arial, sans-serif;
                color: var(--navy);
                background: var(--white);
            }}

            .header {{
                padding: 26px 34px;
                font-size: 52px;
                font-weight: 800;
                border-bottom: 3px solid var(--border);
            }}

            .container {{
                display: grid;
                grid-template-columns: 460px 1fr;
                height: calc(100vh - 112px);
            }}

            .left {{
                border-right: 3px solid var(--border);
                display: flex;
                flex-direction: column;
                background: white;
            }}

            .nav-list {{
                flex: 1;
                background: white;
            }}

            .nav-link {{
                display: block;
                text-decoration: none;
                color: var(--navy);
            }}

            .nav-item {{
                padding: 26px 18px;
                font-size: 34px;
                font-weight: 800;
                text-align: center;
                background: white;
                border-bottom: 3px solid var(--border);
            }}

            .nav-item.selected {{
                background: var(--selected);
            }}

            .left-bottom {{
                border-top: 3px solid var(--border);
                padding: 18px 22px;
                background: white;
            }}

            .back-home {{
                text-decoration: none;
                font-size: 18px;
                font-weight: 700;
                color: var(--navy);
            }}

            .right {{
                padding: 28px 34px;
            }}

            .row {{
                display: grid;
                grid-template-columns: 260px 1fr;
                align-items: center;
                gap: 22px;
                margin: 24px 0 44px 0;
                max-width: 980px;
            }}

            .label {{
                font-size: 40px;
                font-weight: 900;
            }}

            .tri {{
                max-width: 860px;
                position: relative;
            }}

            .tri input[type="radio"] {{
                position: absolute;
                left: -9999px;
            }}

            .track {{
                position: relative;
                height: 44px;
                margin: 0 10px;
                cursor: pointer;
                user-select: none;
            }}

            .line {{
                position: absolute;
                left: 18px;
                right: 18px;
                top: 22px;
                border-top: 3px solid var(--border);
                height: 0;
            }}

            .tick {{
                position: absolute;
                top: 12px;
                width: 3px;
                height: 20px;
                background: var(--border);
            }}

            .tick.t0 {{ left: 18px; }}
            .tick.t1 {{ left: 50%; transform: translateX(-50%); }}
            .tick.t2 {{ right: 18px; }}

            .dot {{
                position: absolute;
                top: 12px;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #d40000;
                transform: translateX(-50%);
                left: 50%;
            }}

            .ticks {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                margin-top: 10px;
                font-size: 22px;
                font-weight: 800;
            }}

            .ticks label {{
                cursor: pointer;
                user-select: none;
            }}

            .ticks label:nth-child(1) {{ justify-self: start; }}
            .ticks label:nth-child(2) {{ justify-self: center; }}
            .ticks label:nth-child(3) {{ justify-self: end; }}

            .toggle-group {{
                display: inline-flex;
                border: 3px solid var(--border);
                border-radius: 14px;
                overflow: hidden;
                width: fit-content;
            }}

            .toggle-group input[type="radio"] {{
                position: absolute;
                left: -9999px;
            }}

            .toggle-option {{
                min-width: 180px;
                padding: 18px 28px;
                font-size: 28px;
                font-weight: 800;
                text-align: center;
                cursor: pointer;
                user-select: none;
                background: var(--white);
                color: var(--navy);
                border-right: 3px solid var(--border);
            }}

            .toggle-option:last-of-type {{
                border-right: none;
            }}

            .toggle-group input[type="radio"]:checked + .toggle-option {{
                background: var(--selected);
            }}

            .cue-selection-panel {{
                max-width: 980px;
                margin-top: 10px;
            }}

            .section-title {{
                font-size: 40px;
                font-weight: 900;
                margin-bottom: 8px;
            }}

            .section-description {{
                font-size: 20px;
                font-weight: 600;
                opacity: 0.85;
                margin-bottom: 24px;
            }}

            .checkbox-group {{
                display: flex;
                flex-wrap: wrap;
                gap: 16px;
            }}

            .checkbox-group input[type="checkbox"] {{
                position: absolute;
                left: -9999px;
            }}

            .checkbox-option {{
                min-width: 180px;
                padding: 18px 28px;
                font-size: 26px;
                font-weight: 800;
                text-align: center;
                cursor: pointer;
                user-select: none;
                background: var(--white);
                color: var(--navy);
                border: 3px solid var(--border);
                border-radius: 14px;
                box-sizing: border-box;
            }}

            .checkbox-group input[type="checkbox"]:checked + .checkbox-option {{
                background: var(--selected);
            }}

            .status {{
                margin-top: 14px;
                font-size: 16px;
                font-weight: 700;
                opacity: 0.85;
            }}

            .tbd {{
                font-size: 28px;
                font-weight: 800;
                margin-top: 10px;
            }}
        </style>
    </head>

    <body>
        <div class="header">Setting</div>

        <div class="container">
            <div class="left">
                <div class="nav-list">__LEFT_NAV__</div>

                <div class="left-bottom">
                    <a class="back-home" href="/">← Back to Homepage</a>
                </div>
            </div>

            <div class="right">
                __RIGHT__
                <div class="status" id="saveStatus"></div>
            </div>
        </div>

        <script>
            const CURRENT_SETTINGS = {current_settings_json};

            const TRI_CONFIG = {{
                font: {{
                    groupName: "font_size",
                    values: ["small", "medium", "large"],
                    defaultValue: CURRENT_SETTINGS.font_size || "medium"
                }},
                image: {{
                    groupName: "image_size",
                    values: ["small", "medium", "large"],
                    defaultValue: CURRENT_SETTINGS.image_size || "medium"
                }},
                duration: {{
                    groupName: "duration_time",
                    values: ["short", "medium", "long"],
                    defaultValue: CURRENT_SETTINGS.duration_time || "medium"
                }}
            }};

            function setDot(triEl, value) {{
                const dot = triEl.querySelector(".dot");
                if (!dot) return;

                const kind = triEl.dataset.kind;
                const cfg = TRI_CONFIG[kind];
                if (!cfg) return;

                const idx = cfg.values.indexOf(value);

                if (idx === 0) dot.style.left = "18px";
                else if (idx === 1) dot.style.left = "50%";
                else dot.style.left = "calc(100% - 18px)";
            }}

            function getChecked(rootEl, groupName, defaultValue) {{
                const checked = rootEl.querySelector(`input[name="${{groupName}}"]:checked`);
                return checked ? checked.value : defaultValue;
            }}

            function getCueSelections() {{
                const checked = document.querySelectorAll('input[name="cue_selection"]:checked');
                if (checked.length > 0) {{
                    return Array.from(checked).map((el) => el.value);
                }}
                return Array.isArray(CURRENT_SETTINGS.cue_selection) && CURRENT_SETTINGS.cue_selection.length
                    ? CURRENT_SETTINGS.cue_selection
                    : ["name", "relationship", "image", "audio"];
            }}

            async function saveSettings(payload) {{
                const status = document.getElementById("saveStatus");
                if (status) status.textContent = "Saving...";

                try {{
                    const res = await fetch("/setting/api/settings", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: JSON.stringify(payload)
                    }});

                    if (!res.ok) {{
                        if (status) status.textContent = "Save failed.";
                        return;
                    }}

                    const data = await res.json();
                    if (!data.ok) {{
                        if (status) status.textContent = "Save failed.";
                        return;
                    }}

                    Object.assign(CURRENT_SETTINGS, payload);

                    if (status) status.textContent = "Saved.";
                    setTimeout(() => {{
                        if (status && status.textContent === "Saved.") status.textContent = "";
                    }}, 1200);

                }} catch (e) {{
                    if (status) status.textContent = "Save failed.";
                    console.error(e);
                }}
            }}

            function snapValueFromClick(trackEl, values, clientX) {{
                const rect = trackEl.getBoundingClientRect();
                const x = clientX - rect.left;
                const ratio = x / rect.width;

                if (ratio < 0.33) return values[0];
                if (ratio > 0.66) return values[2];
                return values[1];
            }}

            async function saveAllCurrentSettings() {{
                const fontTri = document.querySelector('.tri[data-kind="font"]');
                const imgTri = document.querySelector('.tri[data-kind="image"]');
                const durTri = document.querySelector('.tri[data-kind="duration"]');

                const fontV = fontTri
                    ? getChecked(fontTri, TRI_CONFIG.font.groupName, CURRENT_SETTINGS.font_size || TRI_CONFIG.font.defaultValue)
                    : (CURRENT_SETTINGS.font_size || TRI_CONFIG.font.defaultValue);

                const imgV = imgTri
                    ? getChecked(imgTri, TRI_CONFIG.image.groupName, CURRENT_SETTINGS.image_size || TRI_CONFIG.image.defaultValue)
                    : (CURRENT_SETTINGS.image_size || TRI_CONFIG.image.defaultValue);

                const durV = durTri
                    ? getChecked(durTri, TRI_CONFIG.duration.groupName, CURRENT_SETTINGS.duration_time || TRI_CONFIG.duration.defaultValue)
                    : (CURRENT_SETTINGS.duration_time || TRI_CONFIG.duration.defaultValue);

                const voiceV = getChecked(document, "voice_type", CURRENT_SETTINGS.voice_type || "female");
                const cueSelections = getCueSelections();

                await saveSettings({{
                    font_size: fontV,
                    image_size: imgV,
                    duration_time: durV,
                    voice_type: voiceV,
                    cue_selection: cueSelections
                }});
            }}

            function wireTri(triEl) {{
                const kind = triEl.dataset.kind;
                const cfg = TRI_CONFIG[kind];
                if (!cfg) return;

                const groupName = cfg.groupName;
                const track = triEl.querySelector(".track");
                if (!track) return;

                setDot(triEl, getChecked(triEl, groupName, cfg.defaultValue));

                track.addEventListener("click", async (ev) => {{
                    const value = snapValueFromClick(track, cfg.values, ev.clientX);

                    const input = triEl.querySelector(`input[name="${{groupName}}"][value="${{value}}"]`);
                    if (input) input.checked = true;

                    setDot(triEl, value);
                    await saveAllCurrentSettings();
                }});

                triEl.querySelectorAll("label").forEach((lab) => {{
                    lab.addEventListener("click", () => {{
                        setTimeout(async () => {{
                            const v = getChecked(triEl, groupName, cfg.defaultValue);
                            setDot(triEl, v);
                            await saveAllCurrentSettings();
                        }}, 0);
                    }});
                }});
            }}

            function wireVoiceToggle() {{
                const voiceInputs = document.querySelectorAll('input[name="voice_type"]');
                voiceInputs.forEach((input) => {{
                    input.addEventListener("change", async () => {{
                        await saveAllCurrentSettings();
                    }});
                }});
            }}

            function wireCueSelection() {{
                const cueInputs = document.querySelectorAll('input[name="cue_selection"]');
                cueInputs.forEach((input) => {{
                    input.addEventListener("change", async () => {{
                        await saveAllCurrentSettings();
                    }});
                }});
            }}

            document.addEventListener("DOMContentLoaded", () => {{
                const fontTri = document.querySelector('.tri[data-kind="font"]');
                const imgTri = document.querySelector('.tri[data-kind="image"]');
                const durTri = document.querySelector('.tri[data-kind="duration"]');

                if (fontTri) wireTri(fontTri);
                if (imgTri) wireTri(imgTri);
                if (durTri) wireTri(durTri);

                wireVoiceToggle();
                wireCueSelection();
            }});
        </script>
    </body>
    </html>
    """

    html = html.replace("__LEFT_NAV__", left_nav_html).replace("__RIGHT__", right_html)
    return html