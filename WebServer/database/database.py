from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from WebServer.database.add_new_person import router as add_person_router
from WebServer.database.edit_information import router as edit_information_router
from WebServer.database.add_new_person import DEFAULT_DB_DIR, ensure_db_layout

from WebServer._icons_.icon import PENCIL_ICON_SVG, PERSON_ICON_SVG

router = APIRouter()
router.include_router(add_person_router)
router.include_router(edit_information_router)

DEMO_MODE = False  # True -> use FAKE_LOCAL_DB, False -> read people.json

# Fake Local Database 
FAKE_LOCAL_DB = [
    {"id": 1, "headshot": "local/headshots/rachel.mp4", "image": None, "name": "Rachel Kim", "relationship": "Daughter"},
    {"id": 2, "headshot": "local/headshots/sarah.mp4", "image": None, "name": "Sarah Lee", "relationship": "Niece"},
    {"id": 3, "headshot": "local/headshots/chris.mp4", "image": None, "name": "Chris Park", "relationship": "Physician"},
    {"id": 4, "headshot": "local/headshots/mat.mp4", "image": None, "name": "Mat Kim", "relationship": "Son"},
    {"id": 5, "headshot": "local/headshots/joon.mp4", "image": None, "name": "Joon Lee", "relationship": "Niece"},
    {"id": 6, "headshot": "local/headshots/Norah.mp4", "image": None, "name": "Norah Park", "relationship": "Physician"},
    {"id": 7, "headshot": "local/headshots/John.mp4", "image": None, "name": "John Kim", "relationship": "Daughter"},
    {"id": 8, "headshot": "local/headshots/Jane.mp4", "image": None, "name": "Jane Lee", "relationship": "Niece"},
    {"id": 9, "headshot": "local/headshots/Kate.mp4", "image": None, "name": "Kate Park", "relationship": "Physician"},
]

PLACEHOLDER_IMG_DATA_URI = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='600' height='400' viewBox='0 0 600 400'>"
    "<rect width='600' height='400' fill='white'/>"
    "<rect x='70' y='50' width='460' height='300' fill='none' stroke='%230b2a5b' stroke-width='14'/>"
    "<circle cx='175' cy='140' r='18' fill='none' stroke='%230b2a5b' stroke-width='10'/>"
    "<path d='M170 285 L300 120 L430 285 Z' fill='none' stroke='%230b2a5b' stroke-width='14'/>"
    "<path d='M255 210 L300 170 L345 220' fill='none' stroke='%230b2a5b' stroke-width='14'/>"
    "</svg>"
)


def get_people_from_local_db() -> list[dict]:
    if DEMO_MODE:
        return FAKE_LOCAL_DB

    ensure_db_layout(DEFAULT_DB_DIR)
    people_path = DEFAULT_DB_DIR / "people.json"
    try:
        return json.loads(people_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def to_image_url(image_rel: str | None) -> str:
    if not image_rel:
        return PLACEHOLDER_IMG_DATA_URI
    return "/localdb/" + str(image_rel).replace("\\", "/")


@router.get("/database", response_class=HTMLResponse)
def database_page():
    people = get_people_from_local_db()

    people_list_items = []
    for p in people:
        img_src = to_image_url(p.get("image"))
        name = (p.get("name") or "").replace('"', "&quot;")
        rel = (p.get("relationship") or "").replace('"', "&quot;")

        people_list_items.append(
            f"""
            <button
                type="button"
                class="person-row"
                onclick="selectPerson(this)"
                data-id="{p.get('id')}"
                data-name="{name}"
                data-relationship="{rel}"
                data-image="{img_src}"
            >
                <div class="person-icon">{PERSON_ICON_SVG}</div>
                <div class="person-text">
                    <div class="person-line"><span class="label">Name:</span> {name}</div>
                    <div class="person-line"><span class="label">Relationship:</span> {rel}</div>
                </div>
            </button>
            """
        )

    people_list_html = "\n".join(people_list_items)

    return f"""
    <html>
    <head>
        <title>Database</title>
        <style>
            :root {{
                --navy: #0b2a5b;
                --border: #1f4f7a;
                --selected: #aeb7c7;
                --white: #ffffff;
                --disabled: #93a3bd;
                --danger: #b3261e;
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

            /* LEFT PANEL */
            .left {{
                border-right: 3px solid var(--border);
                display: flex;
                flex-direction: column;
                height: 100%;
                min-height: 0;
            }}

            .left-top {{
                border-bottom: 3px solid var(--border);
            }}

            .left-title {{
                font-size: 30px;
                font-weight: 800;
                padding: 18px 22px;
                border-bottom: 3px solid var(--border);
            }}

            .add-row {{
                display: flex;
                align-items: center;
                gap: 18px;
                padding: 18px 22px;
                cursor: pointer;
                text-decoration: none;
                color: var(--navy);
            }}

            .add-circle {{
                width: 52px;
                height: 52px;
                border-radius: 999px;
                background: var(--navy);
                display: flex;
                justify-content: center;
                align-items: center;
                color: white;
                font-size: 34px;
                font-weight: 900;
                line-height: 1;
            }}

            .add-text {{
                font-size: 24px;
                font-weight: 800;
            }}

            .people-list {{
                overflow-y: auto;
                flex: 1;
                min-height: 0;
            }}

            .person-row {{
                width: 100%;
                text-align: left;
                border: none;
                outline: none;

                display: grid;
                grid-template-columns: 92px 1fr;
                align-items: center;
                gap: 10px;

                padding: 18px 16px;
                border-bottom: 3px solid var(--border);

                color: var(--navy);
                background: white;
                cursor: pointer;
            }}

            .person-row.selected {{
                background: var(--selected);
            }}

            .person-icon {{
                display: flex;
                justify-content: center;
                align-items: center;
                color: var(--navy);
            }}

            .person-text {{
                font-size: 22px;
                line-height: 1.4;
            }}

            .label {{
                font-weight: 800;
            }}

            .left-bottom {{
                border-top: 3px solid var(--border);
                padding: 18px 22px;
            }}

            .back-home {{
                text-decoration: none;
                font-size: 18px;
                font-weight: 700;
                color: var(--navy);
            }}

            /* RIGHT PANEL */
            .right {{
                position: relative;
                padding: 26px 34px;
            }}

            .right-empty {{
                height: 100%;
            }}

            /* Edit button */
            .edit-btn {{
                position: absolute;
                top: 26px;
                right: 34px;
                display: none;
                align-items: center;
                gap: 12px;
                padding: 12px 18px;
                border: 2px solid var(--navy);
                color: var(--navy);
                text-decoration: none;
                font-weight: 800;
                font-size: 22px;
                background: white;
                cursor: pointer;
            }}

            .edit-icon {{
                display: inline-flex;
                align-items: center;
            }}

            /* Editing label (edit mode) */
            .editing-label {{
                position: absolute;
                top: 26px;
                left: 34px;
                display: none;
                font-size: 26px;
                font-weight: 900;
                border: 2px solid var(--navy);
                padding: 8px 18px;
                background: white;
            }}

            /* Edit mode buttons */
            .edit-actions {{
                position: absolute;
                right: 34px;
                bottom: 26px;
                display: none;
                gap: 14px;
                align-items: center;
            }}

            .btn {{
                border: none;
                font-weight: 900;
                font-size: 24px;
                padding: 12px 28px;
                cursor: pointer;
            }}

            .btn.save {{
                background: var(--navy);
                color: white;
                opacity: 0.6;
                cursor: not-allowed;
            }}

            .btn.save.enabled {{
                opacity: 1;
                cursor: pointer;
            }}

            .btn.cancel {{
                background: white;
                color: var(--navy);
                border: 2px solid var(--navy);
            }}

            .btn.delete {{
                background: var(--danger);
                color: white;
            }}

            /* VIEW LAYOUT */
            .detail-wrap {{
                margin-top: 90px;
                display: none;
                flex-direction: column;
                align-items: center;
                gap: 24px;
            }}

            .detail-image {{
                width: auto;
                height: 360px;
                object-fit: cover;
                border: 3px solid var(--navy);
            }}

            .detail-lines {{
                font-size: 34px;
                line-height: 1.6;
            }}

            .detail-label {{
                font-weight: 900;
            }}

            /* Image (edit mode) */
            .image-row {{
                display: flex;
                align-items: flex-end;
                justify-content: center;
                gap: 22px;
                margin: 0;
                padding: 0;
            }}

            .change-image-link {{
                font-size: 22px;
                font-weight: 800;
                text-decoration: underline;
                cursor: pointer;
                color: var(--navy);
                white-space: nowrap;
                display: none; /* shown only in edit mode */
                margin-bottom: 10px;
            }}

            /* Inputs */
            .inline-input {{
                font-size: 30px;
                padding: 6px 10px;
                border: 2px solid var(--navy);
                color: var(--navy);
                font-weight: 700;
                vertical-align: middle;
            }}
            .inline-input.name {{ width: 320px; }}
            .inline-input.rel  {{ width: 360px; }}
        </style>
    </head>

    <body>
        <div class="header">Database</div>

        <div class="container">
            <div class="left">
                <div class="left-top">
                    <div class="left-title">List of People in Database</div>

                    <a class="add-row" href="/database/add">
                        <div class="add-circle">+</div>
                        <div class="add-text">Add New Person</div>
                    </a>
                </div>

                <div class="people-list" id="peopleList">
                    {people_list_html}
                </div>

                <div class="left-bottom">
                    <a class="back-home" href="/">← Back to Homepage</a>
                </div>
            </div>

            <div class="right">
                <!-- View-mode edit button -->
                <button class="edit-btn" id="editBtn" type="button">
                    <span class="edit-icon">{PENCIL_ICON_SVG}</span>
                    <span>Edit Information</span>
                </button>

                <!-- Edit-mode label -->
                <div class="editing-label" id="editingLabel">Editing Information</div>

                <!-- Hidden file input -->
                <input id="editImageFile" type="file" accept="image/*" style="display:none;" />

                <!-- Edit-mode bottom-right actions -->
                <div class="edit-actions" id="editActions">
                    <button class="btn delete" id="deleteBtn" type="button">Delete</button>
                    <button class="btn cancel" id="cancelBtn" type="button">Cancel</button>
                    <button class="btn save" id="saveBtn" type="button" disabled>Save</button>
                </div>

                <!-- Single details layout -->
                <div class="detail-wrap" id="detailWrap">
                    <div class="image-row">
                        <img class="detail-image" id="detailImage" src="" alt="person image" />
                        <label class="change-image-link" id="changeImageLink" for="editImageFile">
                            Click to change image
                        </label>
                    </div>

                    <div class="detail-lines">
                        <div class="detail-line">
                            <span class="detail-label">Name:</span>
                            <span id="detailName"></span>
                            <input class="inline-input name" id="editName" type="text" style="display:none;" />
                        </div>
                        <div class="detail-line">
                            <span class="detail-label">Relationship:</span>
                            <span id="detailRelationship"></span>
                            <input class="inline-input rel" id="editRelationship" type="text" style="display:none;" />
                        </div>
                    </div>
                </div>

                <div class="right-empty" id="rightEmpty"></div>
            </div>
        </div>

        <script>
          let selectedId = null;
          let isEditing = false;

          // Originals
          let originalName = "";
          let originalRelationship = "";
          let originalImageSrc = "";
          let imageChanged = false;

          const editBtn = document.getElementById("editBtn");
          const editingLabel = document.getElementById("editingLabel");
          const editActions = document.getElementById("editActions");
          const cancelBtn = document.getElementById("cancelBtn");
          const deleteBtn = document.getElementById("deleteBtn");
          const saveBtn = document.getElementById("saveBtn");
          const changeImageLink = document.getElementById("changeImageLink");
          const editImageFile = document.getElementById("editImageFile");

          const detailWrap = document.getElementById("detailWrap");
          const rightEmpty = document.getElementById("rightEmpty");

          const detailImage = document.getElementById("detailImage");
          const detailName = document.getElementById("detailName");
          const detailRelationship = document.getElementById("detailRelationship");

          const editName = document.getElementById("editName");
          const editRelationship = document.getElementById("editRelationship");

          function setSaveEnabled(enabled) {{
            saveBtn.disabled = !enabled;
            if (enabled) {{
              saveBtn.classList.add("enabled");
              saveBtn.style.cursor = "pointer";
            }} else {{
              saveBtn.classList.remove("enabled");
              saveBtn.style.cursor = "not-allowed";
            }}
          }}

          function isValid() {{
            return editName.value.trim().length > 0 && editRelationship.value.trim().length > 0;
          }}

          function hasChanges() {{
            const nameChanged = editName.value.trim() !== originalName;
            const relChanged = editRelationship.value.trim() !== originalRelationship;
            return nameChanged || relChanged || imageChanged;
          }}

          function refreshSaveLock() {{
            const ok = isValid() && hasChanges();
            setSaveEnabled(ok);
          }}

          function enterEditMode() {{
            if (!selectedId) return;

            isEditing = true;
            imageChanged = false;

            editBtn.style.display = "none";
            editingLabel.style.display = "inline-block";
            editActions.style.display = "flex";
            changeImageLink.style.display = "inline-block";

            detailName.style.display = "none";
            detailRelationship.style.display = "none";
            editName.style.display = "inline-block";
            editRelationship.style.display = "inline-block";

            originalName = (detailName.textContent || "").trim();
            originalRelationship = (detailRelationship.textContent || "").trim();
            originalImageSrc = detailImage.src || "";

            editName.value = originalName;
            editRelationship.value = originalRelationship;

            editImageFile.value = "";
            refreshSaveLock();
          }}

          function exitEditMode() {{
            isEditing = false;
            imageChanged = false;

            editingLabel.style.display = "none";
            editActions.style.display = "none";
            changeImageLink.style.display = "none";
            editBtn.style.display = selectedId ? "inline-flex" : "none";

            editName.style.display = "none";
            editRelationship.style.display = "none";
            detailName.style.display = "inline";
            detailRelationship.style.display = "inline";

            editImageFile.value = "";
            setSaveEnabled(false);
          }}

          function applyViewValues(name, relationship, imageSrc) {{
            detailName.textContent = name;
            detailRelationship.textContent = relationship;
            if (imageSrc) detailImage.src = imageSrc;
          }}

          function selectPerson(buttonEl) {{
            if (isEditing) exitEditMode();

            document.querySelectorAll(".person-row").forEach((el) => el.classList.remove("selected"));
            buttonEl.classList.add("selected");

            selectedId = buttonEl.dataset.id;

            const name = buttonEl.dataset.name || "";
            const relationship = buttonEl.dataset.relationship || "";
            const image = buttonEl.dataset.image || "";

            rightEmpty.style.display = "none";
            detailWrap.style.display = "flex";

            applyViewValues(name, relationship, image);

            editBtn.style.display = "inline-flex";
          }}

          editBtn.addEventListener("click", (e) => {{
            e.preventDefault();
            if (!selectedId) return;
            enterEditMode();
          }});

          editName.addEventListener("input", refreshSaveLock);
          editRelationship.addEventListener("input", refreshSaveLock);

          editImageFile.addEventListener("change", () => {{
            if (!isEditing) return;
            if (!editImageFile.files || editImageFile.files.length === 0) return;

            imageChanged = true;

            const reader = new FileReader();
            reader.onload = (e) => {{
              detailImage.src = e.target.result;
              refreshSaveLock();
            }};
            reader.readAsDataURL(editImageFile.files[0]);
          }});

          cancelBtn.addEventListener("click", () => {{
            if (!isEditing) return;
            applyViewValues(originalName, originalRelationship, originalImageSrc);
            exitEditMode();
          }});

          // DELETE
          deleteBtn.addEventListener("click", async () => {{
            if (!selectedId) return;

            const nm = (detailName.textContent || "").trim();
            const ok = confirm('Delete "' + nm + '" from the database? This cannot be undone.');
            if (!ok) return;

            const fd = new FormData();
            fd.append("person_id", selectedId);

            const res = await fetch("/database/delete", {{
              method: "POST",
              body: fd,
            }});

            if (!res.ok) {{
              const txt = await res.text();
              alert("Delete failed: " + txt);
              return;
            }}

            const data = await res.json();
            if (!data.ok) {{
              alert("Delete failed.");
              return;
            }}

            // Remove selected row from left list
            const selectedBtn = document.querySelector(".person-row.selected");
            if (selectedBtn) selectedBtn.remove();

            // Clear right panel
            selectedId = null;
            exitEditMode();
            detailWrap.style.display = "none";
            rightEmpty.style.display = "block";
          }});

          // SAVE
          saveBtn.addEventListener("click", async () => {{
            if (!isEditing || !selectedId) return;
            if (saveBtn.disabled) return;

            const name = editName.value.trim();
            const relationship = editRelationship.value.trim();

            const fd = new FormData();
            fd.append("person_id", selectedId);
            fd.append("name", name);
            fd.append("relationship", relationship);

            if (editImageFile.files && editImageFile.files.length > 0) {{
              fd.append("image", editImageFile.files[0]);
            }}

            const res = await fetch("/database/update", {{
              method: "POST",
              body: fd,
            }});

            if (!res.ok) {{
              const txt = await res.text();
              alert("Update failed: " + txt);
              return;
            }}

            const data = await res.json();
            if (!data.ok) {{
              alert("Update failed.");
              return;
            }}

            const updated = data.person;

            applyViewValues(updated.name, updated.relationship, updated.image || detailImage.src);

            const selectedBtn = document.querySelector(".person-row.selected");
            if (selectedBtn) {{
              selectedBtn.dataset.name = updated.name;
              selectedBtn.dataset.relationship = updated.relationship;
              if (updated.image) selectedBtn.dataset.image = updated.image;

              selectedBtn.querySelector(".person-text").innerHTML =
                '<div class="person-line"><span class="label">Name:</span> ' + updated.name + '</div>' +
                '<div class="person-line"><span class="label">Relationship:</span> ' + updated.relationship + '</div>';
            }}

            exitEditMode();
          }});
        </script>
    </body>
    </html>
    """