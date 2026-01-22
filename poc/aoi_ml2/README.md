# Magic Leap 2 AOI (gaze + face box) adapter

This folder ports the AOI gaze-on-face logic from `poc/aoi_g3/aoi_engine.py` into a Unity-friendly C# implementation for Magic Leap 2. The focus is on:

- Reusing the dwell/cooldown AOI logic.
- Reading the live camera stream via `MLCamera`.
- Using the ML2 eye tracking API with **MLGazeRecognition_Fixation = 3**.
- Drawing face bounding boxes + gaze point for debugging.

## What moved from the G3 prototype

| G3 prototype | ML2 adapter |
| --- | --- |
| `poc/aoi_g3/aoi_engine.py` | `Scripts/AoiEventEngine.cs` |
| G3 scene camera (RTSP) | `MLCamera` live video stream |
| G3 gaze (gaze2d normalized) | `MLGazeRecognition` fixation gaze point |
| OpenCV face detection | `IAoiFaceDetector` (plug-in detector) |

## Files

- `Scripts/AoiEventEngine.cs`: AOI dwell + cooldown logic ported from the Python AOI engine.
- `Scripts/IAoiFaceDetector.cs`: interface to plug in any face detector that outputs `RectInt` face boxes.
- `Scripts/OpenCvDnnFaceDetector.cs`: **concrete** `IAoiFaceDetector` implementation using the OpenCV DNN SSD face model (requires the OpenCV for Unity plugin).
- `Scripts/ML2GazeAoiController.cs`: glue code that:
  - Starts `MLCamera` and gets live frames.
  - Starts `MLGazeRecognition` in fixation mode (value `3`).
  - Runs AOI logic per frame and logs `AOI_HIT` events (dwell-based or fixation-only).
  - Draws live bounding boxes + gaze point overlay (debug).
- `Scripts/FixationAoiEventEngine.cs`: AOI logic that only fires when `MLGazeRecognition` reports a fixation and the gaze point is inside a face box.

## First-time ML2 test walkthrough (MacBook Pro + Magic Leap 2)

This is a practical, end-to-end checklist for someone new to ML2 and Unity. It assumes basic computer skills but no prior Unity experience.

### 1) Install the tools on your Mac

1. Install **Unity Hub** from Unity’s website.
2. Open Unity Hub and create/sign in with a Unity account (free is fine).
3. In Unity Hub, install a **Unity LTS version supported by Magic Leap 2** (check Magic Leap’s current Unity version support).
4. Install **Magic Leap Hub** (Magic Leap’s desktop companion app) to set up the device and developer mode.
5. (Optional) If you plan to use the OpenCV DNN detector included here, obtain the **OpenCV for Unity** plugin from the Unity Asset Store (paid).

### 2) Set up the Magic Leap 2 headset

1. Power on the headset and run initial setup (Wi-Fi, controller pairing, etc.).
2. Connect the headset to your Mac using USB-C.
3. Open **Magic Leap Hub** and enable **Developer Mode** on the device.
4. Confirm the device shows as connected and authorized.

### 3) Create a Unity project with Magic Leap SDK

1. In Unity Hub, click **New project**.
2. Choose **3D (Core)** and give it a name like `ML2_AOI_Test`, then **Create**.
3. Once the editor opens, install the **Magic Leap SDK** for Unity (follow the official Magic Leap Unity setup guide for your version).
4. In Unity **Project Settings**:
   - Enable the **Magic Leap XR plugin**.
   - Enable **camera** and **eye tracking** permissions.

### 4) Import this AOI adapter

1. In Finder, copy the `poc/aoi_ml2/Scripts` folder into your Unity project under `Assets/`.
2. Back in Unity, wait for the import to finish (progress bar in the bottom-right).
3. (If using OpenCV for Unity) Place the Caffe model files in `Assets/StreamingAssets/`:
   - `deploy.prototxt`
   - `res10_300x300_ssd_iter_140000.caffemodel`

### 5) Build a simple test scene

1. In the **Hierarchy** panel (left), right-click and choose **3D Object > Quad**.
2. With the Quad selected, press **F** to focus it in the Scene view.
3. In the **Hierarchy**, right-click and choose **Create Empty**, name it `AOIController`.
4. With `AOIController` selected, click **Add Component** in the **Inspector** and add `ML2GazeAoiController`.
5. Assign **Camera Renderer**:
   - Select the Quad in the Hierarchy, then drag its **MeshRenderer** component onto the **Camera Renderer** field in `ML2GazeAoiController`.
6. Add a **face detector**:
   - If using OpenCV for Unity, click **Add Component** on `AOIController`, add `OpenCvDnnFaceDetector`, then drag that component into **Face Detector Component**.
7. Set **AOI Mode**:
   - Use **Dwell** if you want dwell + cooldown.
   - Use **Fixation** if you only want fixation-based events.

### 6) Build & deploy to the headset

1. Open **File > Build Settings** and select **Android** as the platform.
2. Click **Switch Platform** (first time can take a few minutes).
3. Ensure your ML2 device is connected (it should appear in the **Run Device** dropdown).
4. Click **Build and Run** and choose a folder for the build output.

### 7) What to look for on device

1. You should see the **live camera feed** on the Quad in the headset.
2. Face boxes should draw when the detector sees faces.
3. When gaze enters a face box:
   - In **Dwell** mode, wait for `minDwellSeconds`.
   - In **Fixation** mode, ML2 must report a fixation.
4. The console should log lines like:
   - `AOI_HIT face=<id> gaze=<x,y> entry=<t> emit=<t>`

### 8) Troubleshooting tips

- **No camera feed**: check Magic Leap permissions (camera access).
- **No gaze**: check eye tracking permissions and confirm `MLGazeRecognition` starts.
- **No face boxes**: verify model files in `StreamingAssets` and that OpenCV for Unity is installed (if using `OpenCvDnnFaceDetector`).

## How to use in Unity (quick setup)

If you already completed the walkthrough above, this section is a shorter checklist you can reuse. It assumes you have a Unity project with the Magic Leap SDK set up.

1. **Copy the scripts into your project**
   - In Finder, drag the folder `poc/aoi_ml2/Scripts` into your Unity project’s `Assets/` folder.
   - Wait for Unity to finish importing (progress bar in the bottom-right).
2. **Create the controller GameObject**
   - In Unity, open the **Hierarchy** panel (left).
   - Right-click inside the Hierarchy and choose **Create Empty**.
   - Name it `AOIController`.
3. **Add the ML2GazeAoiController component**
   - With `AOIController` selected, click **Add Component** in the **Inspector**.
   - Search for `ML2GazeAoiController` and add it.
4. **Add a surface to show the camera feed**
   - In the Hierarchy, right-click → **3D Object > Quad**.
   - Select the Quad and press **F** to focus it in the Scene view.
5. **Wire the camera feed to the controller**
   - Select `AOIController`.
   - In the Inspector for `ML2GazeAoiController`, find **Camera Renderer**.
   - Drag the Quad’s **MeshRenderer** component into that field.
6. **Add a face detector**
   - If you use OpenCV for Unity, select `AOIController`, click **Add Component**, and add `OpenCvDnnFaceDetector`.
   - Drag the `OpenCvDnnFaceDetector` component into **Face Detector Component**.
   - Make sure the model files are in `Assets/StreamingAssets/`:
     - `deploy.prototxt`
     - `res10_300x300_ssd_iter_140000.caffemodel`
7. **Choose the AOI mode**
   - In `ML2GazeAoiController`, set **AOI Mode**:
     - **Dwell** for the original dwell + cooldown behavior.
     - **Fixation** for fixation-only events.
8. **Adjust tuning values**
   - Set `minDwellSeconds` (only used in **Dwell** mode).
   - Set `cooldownSeconds` to avoid repeat triggers too quickly.

> **Note:** Magic Leap API names can vary between SDK versions. If you get compile errors, adjust the calls in `ML2GazeAoiController` to match your installed MLSDK.

## Face detection integration

The AOI engine expects per-frame face boxes in the camera image pixel space. This repo **already includes** one concrete implementation: `OpenCvDnnFaceDetector` (OpenCV DNN SSD). Implement `IAoiFaceDetector` yourself if you want to swap in a different detector.

Implement `IAoiFaceDetector` using one of the following:

- OpenCV (Unity plugin) with a lightweight face detector.
- ML2 CV APIs if your build already uses them.
- An on-device model inference pipeline.

Return a dictionary mapping stable face IDs to `RectInt` boxes so the AOI engine can track dwell time per face.

## Fixation mode

`ML2GazeAoiController` sets `MLGazeRecognition` to fixation mode using the enum value `3`, which maps to `MLGazeRecognition_Fixation` in the ML2 API documentation. This ensures AOI events are only generated when a fixation is recognized.
