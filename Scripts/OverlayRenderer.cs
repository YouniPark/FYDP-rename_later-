using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class OverlayRenderer : MonoBehaviour
{
    [Header("Providers")]
    public MLCameraFrameProvider cameraProvider;
    public FaceDetectorSentis faceDetector;
    public EyeTrackerProvider eyeTracker;                 
    public GazeAndCameraPermissionManager perms;          
    public FaceFixationTracker fixationTracker;

    [Header("UI")]
    public RawImage cameraRawImage;
    public RectTransform overlayRoot;
    public RectTransform gazeDot;
    public TMP_Text statusText;
    public Image faceBoxPrefab;

    private readonly List<FaceDetectorSentis.Detection> _faces = new List<FaceDetectorSentis.Detection>(32);
    private readonly List<Image> _boxPool = new List<Image>(32);

    private MLCameraFrameProvider.CameraFrame _latestFrame;
    private Vector2 _latestGazePixel;
    private bool _latestGazeInFrame;
    private float _nextLogTime;

    private void Update()
    {
        if (cameraProvider == null || faceDetector == null || fixationTracker == null) return;
        if (cameraRawImage == null || overlayRoot == null) return;

        if (!cameraProvider.TryGetLatestFrame(out _latestFrame)) return;
        if (_latestFrame.Texture == null) return;

        cameraRawImage.texture = _latestFrame.Texture;

        faceDetector.SetSourceTexture(_latestFrame.Texture);

        _faces.Clear();
        if (faceDetector.LatestDetections != null)
            _faces.AddRange(faceDetector.LatestDetections);

        bool sampleValid = false;
        bool projected = false;

        if (perms != null && perms.EyeTrackingPermissionGranted && perms.HasGazePose)
        {
            sampleValid = true;
            projected = GazeToImageProjector.ProjectRayToPixel(
                perms.GazeOriginWorld,
                perms.GazeDirectionWorld,
                _latestFrame.CameraPoseWorld,
                _latestFrame.Intrinsics,
                _latestFrame.Width,
                _latestFrame.Height,
                out _latestGazePixel
            );
        }
        else
        {
            if (eyeTracker != null)
            {
                var gaze = eyeTracker.LatestSample;
                sampleValid = gaze.IsValid;

                if (sampleValid)
                {
                    projected = GazeToImageProjector.ProjectRayToPixel(
                        gaze.OriginWorld,
                        gaze.DirectionWorld,
                        _latestFrame.CameraPoseWorld,
                        _latestFrame.Intrinsics,
                        _latestFrame.Width,
                        _latestFrame.Height,
                        out _latestGazePixel
                    );
                }
            }
        }

        _latestGazeInFrame = sampleValid && projected;

        if (!_latestGazeInFrame)
            _latestGazePixel = new Vector2(_latestFrame.Width * 0.5f, _latestFrame.Height * 0.5f);

        if (Time.time >= _nextLogTime)
        {
            _nextLogTime = Time.time + 0.25f;

            string src =
                (perms != null && perms.EyeTrackingPermissionGranted && perms.HasGazePose) ? "perms" :
                (eyeTracker != null ? "eyeTracker" : "none");

            if (eyeTracker != null)
            {
                var g = eyeTracker.LatestSample;
                Debug.Log($"[Eye] valid={g.IsValid} calibrated={g.IsCalibrated} ts={g.TimestampNs}");
            }

            Debug.Log(
                $"[Overlay] src={src} sampleValid={sampleValid} projected={projected} gazeInFrame={_latestGazeInFrame} " +
                $"gazePix=({_latestGazePixel.x:0.0},{_latestGazePixel.y:0.0}) " +
                $"frame=({_latestFrame.Width}x{_latestFrame.Height}) intrValid={_latestFrame.Intrinsics.IsValid} " +
                $"faces={_faces.Count}"
            );
        }

       
        long ts = 0;
        bool fixEvent = false;
        if (eyeTracker != null)
        {
            var gaze = eyeTracker.LatestSample;
            ts = gaze.TimestampNs;
            fixEvent = gaze.IsFixatingEvent;
        }
        else
        {
            ts = (long)(Time.realtimeSinceStartup * 1e9);
            fixEvent = false;
        }

        fixationTracker.UpdateFixation(
            _latestGazePixel,
            _latestGazeInFrame,
            ts,
            _faces,
            fixEvent
        );

        DrawFaceBoxes();
        DrawGazeDot();
        UpdateStatus();
    }

    private void DrawFaceBoxes()
    {
        if (faceBoxPrefab == null || overlayRoot == null) return;

        EnsurePool(_faces.Count);

        for (int i = 0; i < _boxPool.Count; i++)
        {
            bool active = i < _faces.Count;
            _boxPool[i].gameObject.SetActive(active);
            if (!active) continue;

            var rect = _faces[i].PixelRect;
            SetUiRectFromPixelRect(_boxPool[i].rectTransform, rect);
        }
    }

    private void DrawGazeDot()
    {
        if (gazeDot == null || overlayRoot == null) return;

        gazeDot.gameObject.SetActive(true);

        float xNorm = _latestGazePixel.x / Mathf.Max(1, _latestFrame.Width);
        float yNorm = _latestGazePixel.y / Mathf.Max(1, _latestFrame.Height);

        var size = overlayRoot.rect.size;
        gazeDot.anchoredPosition = new Vector2((xNorm - 0.5f) * size.x, (yNorm - 0.5f) * size.y);
    }

    private void UpdateStatus()
    {
        if (statusText == null) return;

        bool permsEye = (perms != null && perms.EyeTrackingPermissionGranted);
        bool permsCam = (perms != null && perms.CameraPermissionGranted);

        statusText.text =
            $"FIXATING ON FACE: {fixationTracker.IsFixatingOnFace}\n" +
            $"Dwell (ms): {fixationTracker.CurrentDwellMs}\n" +
            $"Faces: {_faces.Count}\n" +
            $"Perms Eye: {permsEye}\n" +
            $"Perms Cam: {permsCam}\n" +
            $"Intrinsics exact: {_latestFrame.Intrinsics.IsValid}\n" +
            $"Detector busy: {faceDetector.IsBusy}";
    }

    private void EnsurePool(int needed)
    {
        while (_boxPool.Count < needed)
        {
            var box = Instantiate(faceBoxPrefab, overlayRoot);
            _boxPool.Add(box);
        }
    }

    private void SetUiRectFromPixelRect(RectTransform target, Rect pixelRect)
    {
        float xMin = pixelRect.xMin / Mathf.Max(1f, _latestFrame.Width);
        float xMax = pixelRect.xMax / Mathf.Max(1f, _latestFrame.Width);
        float yMin = pixelRect.yMin / Mathf.Max(1f, _latestFrame.Height);
        float yMax = pixelRect.yMax / Mathf.Max(1f, _latestFrame.Height);

        var size = overlayRoot.rect.size;
        float left = (xMin - 0.5f) * size.x;
        float right = (xMax - 0.5f) * size.x;
        float bottom = (yMin - 0.5f) * size.y;
        float top = (yMax - 0.5f) * size.y;

        target.anchoredPosition = new Vector2((left + right) * 0.5f, (bottom + top) * 0.5f);
        target.sizeDelta = new Vector2(right - left, top - bottom);
    }
}
