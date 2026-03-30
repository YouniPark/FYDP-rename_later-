from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from WebServer.database.add_new_person import DEFAULT_DB_DIR
from WebServer.database.database import router as database_router
from WebServer.setting.setting import router as setting_router

from WebServer._icons_.icon import DATABASE_ICON_SVG, SETTING_ICON_SVG

app = FastAPI()

app.include_router(database_router)
app.include_router(setting_router)

app.mount("/localdb", StaticFiles(directory=str(DEFAULT_DB_DIR)), name="localdb")


# # Create public URL for image and audio cues 
# app.mount(
#     "/media/images",
#     StaticFiles(directory=DEFAULT_DB_DIR/"images"),
#     name="images",
# )

# app.mount(
#     "/media/audio",
#     StaticFiles(directory=DEFAULT_DB_DIR/"auditory cues"),
#     name="audio",
# )


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
    <html>
        <head>
            <title>Main Page</title>
            <style>
                body {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 5vw;
                    height: 100vh;
                    margin: 0;
                    font-family: Arial;
                    background-color: white;
                }}

                .box {{
                    width: 25vw;
                    height: 25vw;
                    max-width: 320px;
                    max-height: 320px;
                    min-width: 200px;
                    min-height: 200px;

                    background-color: #0b2a5b;
                    color: white;

                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;

                    border-radius: 6px;
                    cursor: pointer;
                    text-align: center;
                }}

                .icon svg {{
                    width: 70px;
                    height: 70px;
                    margin-bottom: 25px;
                    display: block;
                }}

                .label {{
                    font-size: 28px;
                    font-weight: bold;
                }}

                a {{
                    text-decoration: none;
                    color: white;
                }}
            </style>
        </head>

        <body>
            <a href="/database">
                <div class="box">
                    <div class="icon">{DATABASE_ICON_SVG}</div>
                    <div class="label">Database</div>
                </div>
            </a>

            <a href="/setting">
                <div class="box">
                    <div class="icon">{SETTING_ICON_SVG}</div>
                    <div class="label">Setting</div>
                </div>
            </a>
        </body>
    </html>
    """
