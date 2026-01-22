using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.MagicLeap;

namespace AoiMl2
{
    public sealed class ML2GazeAoiController : MonoBehaviour
    {
        private const int MLGazeRecognitionFixation = 3;

        [Header("AOI Settings")]
        [SerializeField] private float minDwellSeconds = 0.10f;
        [SerializeField] private float cooldownSeconds = 0.50f;

        [Header("AOI Mode")]
        [SerializeField] private AoiMode aoiMode = AoiMode.Dwell;

        [Header("Camera Settings")]
        [SerializeField] private Renderer cameraRenderer;
        [SerializeField] private int targetWidth = 1280;
        [SerializeField] private int targetHeight = 720;

        [Header("Face Detector")]
        [SerializeField] private MonoBehaviour faceDetectorComponent;

        [Header("Debug Overlay")]
        [SerializeField] private bool drawDebugOverlay = true;
        [SerializeField] private Color faceBoxColor = Color.green;
        [SerializeField] private Color hitBoxColor = Color.red;
        [SerializeField] private Color gazeColor = Color.cyan;

        private IAoiFaceDetector _faceDetector;
        private AoiEventEngine _aoiEngine;
        private FixationAoiEventEngine _fixationEngine;
        private MLCamera _camera;
        private Texture2D _cameraTexture;
        private IReadOnlyDictionary<int, RectInt> _latestFaces = new Dictionary<int, RectInt>();
        private RectInt? _lastHitBox;
        private Vector2 _lastGazeNormalized;
        private bool _hasGaze;
        private bool _isFixation;

        private void Awake()
        {
            _faceDetector = faceDetectorComponent as IAoiFaceDetector;
            if (_faceDetector == null && faceDetectorComponent != null)
            {
                Debug.LogWarning("Face detector component does not implement IAoiFaceDetector.");
            }

            _aoiEngine = new AoiEventEngine(minDwellSeconds, cooldownSeconds);
            _fixationEngine = new FixationAoiEventEngine(cooldownSeconds);
        }

        private void OnEnable()
        {
            StartCamera();
            StartGazeRecognition();
        }

        private void OnDisable()
        {
            StopCamera();
            StopGazeRecognition();
        }

        private void Update()
        {
            if (!_hasGaze || _cameraTexture == null)
            {
                return;
            }

            if (_faceDetector != null)
            {
                _latestFaces = _faceDetector.DetectFaces(_cameraTexture);
            }

            var now = Time.time;
            var didHit = aoiMode == AoiMode.Dwell
                ? _aoiEngine.Step(
                    _lastGazeNormalized,
                    _cameraTexture.width,
                    _cameraTexture.height,
                    _latestFaces,
                    now,
                    out var hitEvent)
                : _fixationEngine.Step(
                    _lastGazeNormalized,
                    _cameraTexture.width,
                    _cameraTexture.height,
                    _latestFaces,
                    now,
                    _isFixation,
                    out var hitEvent);

            if (didHit)
            {
                _lastHitBox = hitEvent.Box;
                Debug.Log($"AOI_HIT face={hitEvent.FaceId} gaze={hitEvent.GazeNormalized} entry={hitEvent.EntryTimestamp:F3} emit={hitEvent.EmitTimestamp:F3}");
            }
            else
            {
                _lastHitBox = null;
            }
        }

        private void OnGUI()
        {
            if (!drawDebugOverlay || _cameraTexture == null)
            {
                return;
            }

            var displayRect = new Rect(0, 0, Screen.width, Screen.height);
            GUI.DrawTexture(displayRect, _cameraTexture, ScaleMode.ScaleToFit);

            foreach (var box in _latestFaces.Values)
            {
                DrawRect(box, faceBoxColor);
            }

            if (_lastHitBox.HasValue)
            {
                DrawRect(_lastHitBox.Value, hitBoxColor, thickness: 4);
            }

            if (_hasGaze)
            {
                DrawGazePoint(_lastGazeNormalized, gazeColor);
            }
        }

        private void DrawRect(RectInt rect, Color color, int thickness = 2)
        {
            var scaleX = (float)Screen.width / _cameraTexture.width;
            var scaleY = (float)Screen.height / _cameraTexture.height;
            var scaledRect = new Rect(
                rect.xMin * scaleX,
                rect.yMin * scaleY,
                rect.width * scaleX,
                rect.height * scaleY);

            var prevColor = GUI.color;
            GUI.color = color;
            GUI.DrawTexture(new Rect(scaledRect.x, scaledRect.y, scaledRect.width, thickness), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(scaledRect.x, scaledRect.yMax - thickness, scaledRect.width, thickness), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(scaledRect.x, scaledRect.y, thickness, scaledRect.height), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(scaledRect.xMax - thickness, scaledRect.y, thickness, scaledRect.height), Texture2D.whiteTexture);
            GUI.color = prevColor;
        }

        private void DrawGazePoint(Vector2 gazeNormalized, Color color)
        {
            var px = gazeNormalized.x * Screen.width;
            var py = gazeNormalized.y * Screen.height;
            var size = 12f;
            var prevColor = GUI.color;
            GUI.color = color;
            GUI.DrawTexture(new Rect(px - size / 2f, py - size / 2f, size, size), Texture2D.whiteTexture);
            GUI.color = prevColor;
        }

        private void StartCamera()
        {
            if (!MLCamera.IsSupported)
            {
                Debug.LogError("MLCamera not supported on this device.");
                return;
            }

            var settings = new MLCamera.Settings
            {
                CaptureType = MLCamera.CaptureType.Video,
                ImageFormat = MLCamera.OutputFormat.RGBA_8888,
                Width = targetWidth,
                Height = targetHeight,
                FrameRate = 30
            };

            _camera = MLCamera.CreateAndConnect(settings);
            if (_camera == null)
            {
                Debug.LogError("Failed to connect to MLCamera.");
                return;
            }

            _cameraTexture = new Texture2D(settings.Width, settings.Height, TextureFormat.RGBA32, false);
            _camera.OnRawVideoFrameAvailable += HandleVideoFrame;
            _camera.CaptureVideoStart();

            if (cameraRenderer != null)
            {
                cameraRenderer.material.mainTexture = _cameraTexture;
            }
        }

        private void StopCamera()
        {
            if (_camera == null)
            {
                return;
            }

            _camera.OnRawVideoFrameAvailable -= HandleVideoFrame;
            _camera.CaptureVideoStop();
            _camera.Disconnect();
            _camera = null;
        }

        private void HandleVideoFrame(MLCamera.CameraOutput output)
        {
            if (_cameraTexture == null)
            {
                return;
            }

            if (output?.PlanarData == null || output.PlanarData.Length == 0)
            {
                return;
            }

            var data = output.PlanarData[0].Data;
            if (data == null)
            {
                return;
            }

            _cameraTexture.LoadRawTextureData(data);
            _cameraTexture.Apply();
        }

        private void StartGazeRecognition()
        {
            if (!MLGazeRecognition.IsSupported)
            {
                Debug.LogError("MLGazeRecognition not supported on this device.");
                return;
            }

            var settings = MLGazeRecognition.Settings.Default;
            settings.RecognitionMode = (MLGazeRecognition.Mode)MLGazeRecognitionFixation;
            MLGazeRecognition.Start(settings);
        }

        private void StopGazeRecognition()
        {
            if (MLGazeRecognition.IsSupported)
            {
                MLGazeRecognition.Stop();
            }
        }

        private void LateUpdate()
        {
            if (!MLGazeRecognition.IsSupported || !MLGazeRecognition.IsStarted)
            {
                _hasGaze = false;
                return;
            }

            if (MLGazeRecognition.GetGazeRay(out var gazeRay, out var gazeState))
            {
                if ((int)gazeState.RecognitionMode == MLGazeRecognitionFixation)
                {
                    var normalized = new Vector2(
                        Mathf.Clamp01(gazeState.GazePoint.x),
                        Mathf.Clamp01(gazeState.GazePoint.y));
                    _lastGazeNormalized = normalized;
                    _hasGaze = true;
                    _isFixation = true;
                }
                else
                {
                    _hasGaze = false;
                    _isFixation = false;
                }
            }
            else
            {
                _hasGaze = false;
                _isFixation = false;
            }
        }

        private enum AoiMode
        {
            Dwell,
            Fixation
        }
    }
}
