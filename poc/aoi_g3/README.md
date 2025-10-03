## FOLDER CONTENTS

aoi_g3/
├─ __init__.py
├─ config.py
├─ utils.py
├─ face_model.py
├─ detector.py
├─ tracker.py
├─ markers.py
├─ aoi_engine.py
├─ g3_streamer.py
└─ app.py

## STEPS TO RUN FOLDER

1. Put `aoi_g3` folder on PYTHONPATH (or `pip install -e` if a package)

2. Install the following: 
- `opencv-python`
- `numpy`
- `pylsl`
- `g3pylib`
    
    This can be done by running `pip install -r requirements.txt` from within the aoi_g3 directory.

    To install g3pylib, follow the instructions here: https://github.com/tobii/glasses3-pylib.

3. Edit HOSTNAME in aoi_g3/app.py (serial or IP).

4. Run in terminal: `python -m aoi_g3.app`.
