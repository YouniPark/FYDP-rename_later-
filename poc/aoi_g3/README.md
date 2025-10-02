FOLDER CONTENTS

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

STEPS TO RUN FOLDER

1. put aoi_g3 folder on PYTHONPATH (or pip install -e if a package)

2. install the following: 
pip install opencv-python numpy pylsl g3pylib

3. edit HOSTNAME in aoi_g3/app.py (serial or IP)

4. run in terminal: python -m aoi_g3.app
