from __future__ import annotations

import json
import time
from pathlib import Path
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav
from threading import Lock

router = APIRouter()

ALLOW_SKIP_HEADSHOT_FOR_NOW = False
people_file_lock = Lock()


def default_db_dir() -> Path:
    """
    Default DB folder:
      WebServer/database/PeopleDatabase
    """
    webserver_dir = Path(__file__).resolve().parent
    db_dir = webserver_dir / "PeopleDatabase"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


DEFAULT_DB_DIR = default_db_dir()


def ensure_db_layout(db_dir: Path) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "images").mkdir(parents=True, exist_ok=True)
    (db_dir / "headshots").mkdir(parents=True, exist_ok=True)
    (db_dir / "auditory cues").mkdir(parents=True, exist_ok=True)

    people_json = db_dir / "people.json"
    if not people_json.exists():
        people_json.write_text("[]", encoding="utf-8")


def read_people(db_dir: Path) -> list[dict]:
    people_json = db_dir / "people.json"
    try:
        with people_file_lock:
            return json.loads(people_json.read_text(encoding="utf-8"))
    except Exception:
        return []


def write_people(db_dir: Path, people: list[dict]) -> None:
    people_json = db_dir / "people.json"
    with people_file_lock:
        people_json.write_text(json.dumps(people, indent=2), encoding="utf-8")


def next_person_id(people: list[dict]) -> int:
    if not people:
        return 1
    return max(int(p.get("id", 0)) for p in people) + 1


def auditory_cue(
    name: str,
    relationship: str,
    person_id: int,
    ts: int,
    db_dir: Path,
) -> tuple[str, str]:
    """
    Returns relative paths for JSON storage.
    """
    text_prompt = f"This is your {relationship} {name}."

    audio_male = generate_audio(text_prompt, history_prompt="v2/en_speaker_6")
    audio_female = generate_audio(text_prompt, history_prompt="v2/en_speaker_9")

    cues_dir = db_dir / "auditory cues"
    cues_dir.mkdir(parents=True, exist_ok=True)

    male_filename = f"{person_id}_{ts}_male.wav"
    female_filename = f"{person_id}_{ts}_female.wav"

    male_path = cues_dir / male_filename
    female_path = cues_dir / female_filename

    write_wav(male_path, SAMPLE_RATE, audio_male)
    write_wav(female_path, SAMPLE_RATE, audio_female)

    return (
        str(Path("auditory cues") / male_filename),
        str(Path("auditory cues") / female_filename),
    )


def generate_and_store_auditory_cues(
    person_id: int,
    name: str,
    relationship: str,
    ts: int,
    db_dir: Path,
) -> None:
    try:
        male_rel, female_rel = auditory_cue(
            name=name,
            relationship=relationship,
            person_id=person_id,
            ts=ts,
            db_dir=db_dir,
        )

        people = read_people(db_dir)
        for person in people:
            if int(person.get("id", 0)) == person_id:
                person["auditory cue (male)"] = male_rel
                person["auditory cue (female)"] = female_rel
                person["audio_status"] = "ready"
                person["audio_error"] = None
                break
        write_people(db_dir, people)

    except Exception as e:
        people = read_people(db_dir)
        for person in people:
            if int(person.get("id", 0)) == person_id:
                person["audio_status"] = "failed"
                person["audio_error"] = str(e)
                break
        write_people(db_dir, people)


@router.get("/database/add", response_class=HTMLResponse)
def add_new_person_page():
    ensure_db_layout(DEFAULT_DB_DIR)

    db_label = str(DEFAULT_DB_DIR)
    skip_headshot_js = "true" if ALLOW_SKIP_HEADSHOT_FOR_NOW else "false"

    return f"""
    <html>
    <head>
      <title>Add New Person</title>
      <style>
        :root {{
          --navy: #0b2a5b;
          --border: #1f4f7a;
          --ok: #11a84a;
        }}

        body {{
          margin: 0;
          font-family: Arial, sans-serif;
          color: var(--navy);
          background: white;
        }}

        .header {{
          padding: 18px 28px;
          font-size: 40px;
          font-weight: 800;
          border-bottom: 3px solid var(--border);
        }}

        .wrap {{
          padding: 20px 34px;
          max-width: 1200px;
        }}

        .db-info {{
          border: 2px solid var(--border);
          padding: 10px 14px;
          margin-bottom: 16px;
          font-weight: 800;
          font-size: 16px;
        }}

        .db-path {{
          font-weight: 900;
          word-break: break-all;
        }}

        .note {{
          margin-top: 6px;
          font-size: 14px;
          font-weight: 700;
        }}

        .step {{
          margin: 16px 0 20px 0;
        }}

        .step-title {{
          font-size: 26px;
          font-weight: 900;
          display: flex;
          align-items: center;
          gap: 12px;
        }}

        .step-link {{
          text-decoration: underline;
          cursor: pointer;
        }}

        .desc {{
          margin-top: 8px;
          font-size: 18px;
          line-height: 1.35;
          font-weight: 600;
        }}

        .check {{
          width: 28px;
          height: 28px;
          border: 3px solid var(--ok);
          display: none;
          align-items: center;
          justify-content: center;
          color: var(--ok);
          font-weight: 900;
          line-height: 1;
          font-size: 18px;
        }}

        .preview {{
          margin-top: 10px;
          display: none;
        }}

        .preview img {{
          width: 260px;
          height: auto;
          border: 3px solid var(--navy);
        }}

        .input {{
          margin-top: 10px;
          width: 720px;
          max-width: 100%;
          padding: 10px 12px;
          font-size: 18px;
          border: 2px solid var(--border);
          color: var(--navy);
        }}

        .input.relationship {{
          width: 980px;
          max-width: 100%;
        }}

        .footer {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 18px;
        }}

        .save {{
          background: var(--navy);
          color: white;
          border: none;
          font-weight: 900;
          font-size: 20px;
          padding: 10px 28px;
          cursor: pointer;
          opacity: 0.5;
        }}

        .save.enabled {{
          opacity: 1;
          background: var(--navy);
        }}

        .back {{
          text-decoration: none;
          color: var(--navy);
          font-weight: 800;
          font-size: 16px;
        }}

        input[type="file"] {{
          display: none;
        }}
      </style>
    </head>

    <body>
      <div class="header">Add New Person</div>

      <div class="wrap">
        <div class="db-info">
          Database location: <span class="db-path">{db_label}</span>
          <div class="note">
            {"Headshot upload is optional for now." if ALLOW_SKIP_HEADSHOT_FOR_NOW else ""}
          </div>
        </div>

        <form id="addForm" method="post" action="/database/add/save" enctype="multipart/form-data">

          <div class="step">
            <div class="step-title">
              <span>1.</span>
              <span class="step-link" onclick="document.getElementById('headshot').click()">Click to upload a headshot</span>
              <span class="check" id="check1">✓</span>
            </div>
            <div class="desc" id="desc1">
              Upload a video of a person you want to add in the database with a clear background.<br/>
              Video must include: (1) facing camera, (2) head left, (3) head right, (4) head up, (5) head down.
            </div>
            <input id="headshot" name="headshot" type="file" accept="video/*" />
          </div>

          <div class="step">
            <div class="step-title">
              <span>2.</span>
              <span class="step-link" onclick="document.getElementById('image').click()">Click to upload an image</span>
              <span class="check" id="check2">✓</span>
            </div>
            <div class="desc" id="desc2">
              Upload a photo of the person you want to add in the database that was taken with the user.<br/>
              This image will be used as a cue.
            </div>
            <input id="image" name="image" type="file" accept="image/*" />
            <div class="preview" id="imgPreview">
              <img id="imgTag" src="" alt="preview"/>
            </div>
          </div>

          <div class="step">
            <div class="step-title">
              <span>3.</span>
              <span>Add the name</span>
              <span class="check" id="check3">✓</span>
            </div>
            <input class="input" id="name" name="name" type="text"
                   placeholder="Type in the person’s full name (First name Last name)" />
          </div>

          <div class="step">
            <div class="step-title">
              <span>4.</span>
              <span>Add the relationship</span>
              <span class="check" id="check4">✓</span>
            </div>
            <input class="input relationship" id="relationship" name="relationship" type="text"
                   placeholder="Type in the person’s relationship with the user (e.g. Daughter, Caregiver, Physician, etc.)" />
          </div>

          <div class="footer">
            <a class="back" href="/database">← Back to Database</a>
            <button class="save" id="saveBtn" type="submit" disabled>SAVE</button>
          </div>
        </form>
      </div>

      <script>
        const ALLOW_SKIP_HEADSHOT = {skip_headshot_js};

        const headshotEl = document.getElementById("headshot");
        const imageEl = document.getElementById("image");
        const nameEl = document.getElementById("name");
        const relEl = document.getElementById("relationship");
        const saveBtn = document.getElementById("saveBtn");

        const state = {{
          headshotSelected: false,
          imageSelected: false,
          nameFilled: false,
          relationshipFilled: false,
        }};

        function setDone(stepNum, done) {{
          const check = document.getElementById("check" + stepNum);
          const desc = document.getElementById("desc" + stepNum);

          if (stepNum === 1 && ALLOW_SKIP_HEADSHOT) {{
            done = true;
          }}

          check.style.display = done ? "inline-flex" : "none";

          if (desc) {{
            desc.style.display = done ? "none" : "block";
          }}
        }}

        function refresh() {{
          setDone(1, state.headshotSelected);
          setDone(2, state.imageSelected);
          setDone(3, state.nameFilled);
          setDone(4, state.relationshipFilled);

          const ok =
            (ALLOW_SKIP_HEADSHOT ? true : state.headshotSelected) &&
            state.imageSelected &&
            state.nameFilled &&
            state.relationshipFilled;

          saveBtn.disabled = !ok;
          if (ok) saveBtn.classList.add("enabled");
          else saveBtn.classList.remove("enabled");
        }}

        headshotEl.addEventListener("change", () => {{
          state.headshotSelected = !!(headshotEl.files && headshotEl.files.length);
          refresh();
        }});

        imageEl.addEventListener("change", () => {{
          state.imageSelected = !!(imageEl.files && imageEl.files.length);

          if (state.imageSelected) {{
            const file = imageEl.files[0];
            const reader = new FileReader();
            reader.onload = (e) => {{
              document.getElementById("imgTag").src = e.target.result;
              document.getElementById("imgPreview").style.display = "block";
              refresh();
            }};
            reader.readAsDataURL(file);
          }} else {{
            document.getElementById("imgPreview").style.display = "none";
            refresh();
          }}
        }});

        nameEl.addEventListener("input", () => {{
          state.nameFilled = nameEl.value.trim().length > 0;
          refresh();
        }});

        relEl.addEventListener("input", () => {{
          state.relationshipFilled = relEl.value.trim().length > 0;
          refresh();
        }});

        window.addEventListener("load", () => {{
          state.headshotSelected = !!(headshotEl.files && headshotEl.files.length);
          state.imageSelected = !!(imageEl.files && imageEl.files.length);
          state.nameFilled = nameEl.value.trim().length > 0;
          state.relationshipFilled = relEl.value.trim().length > 0;
          refresh();
        }});
      </script>
    </body>
    </html>
    """


@router.post("/database/add/save")
async def add_new_person_save(
    background_tasks: BackgroundTasks,
    headshot: UploadFile | None = File(default=None),
    image: UploadFile = File(...),
    name: str = Form(...),
    relationship: str = Form(...),
):
    ensure_db_layout(DEFAULT_DB_DIR)

    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    if not relationship.strip():
        raise HTTPException(status_code=400, detail="Relationship is required.")
    if image is None:
        raise HTTPException(status_code=400, detail="Image is required.")
    if not ALLOW_SKIP_HEADSHOT_FOR_NOW and headshot is None:
        raise HTTPException(status_code=400, detail="Headshot is required.")

    people = read_people(DEFAULT_DB_DIR)
    person_id = next_person_id(people)
    ts = int(time.time())

    image_ext = Path(image.filename or "").suffix or ".png"
    image_filename = f"{person_id}_{ts}{image_ext}"
    image_path = DEFAULT_DB_DIR / "images" / image_filename
    image_path.write_bytes(await image.read())

    headshot_rel = None
    if headshot is not None:
        headshot_ext = Path(headshot.filename or "").suffix or ".mp4"
        headshot_filename = f"{person_id}_{ts}{headshot_ext}"
        headshot_path = DEFAULT_DB_DIR / "headshots" / headshot_filename
        headshot_path.write_bytes(await headshot.read())
        headshot_rel = str(Path("headshots") / headshot_filename)

    new_person = {
        "id": person_id,
        "headshot": headshot_rel,
        "image": str(Path("images") / image_filename),
        "name": name.strip(),
        "relationship": relationship.strip(),
        "auditory cue (male)": None,
        "auditory cue (female)": None,
        "audio_status": "pending",
        "audio_error": None,
        "created_at": ts,
    }

    people.append(new_person)
    write_people(DEFAULT_DB_DIR, people)

    background_tasks.add_task(
        generate_and_store_auditory_cues,
        person_id,
        name.strip(),
        relationship.strip(),
        ts,
        DEFAULT_DB_DIR,
    )

    return RedirectResponse(url="/database", status_code=303)