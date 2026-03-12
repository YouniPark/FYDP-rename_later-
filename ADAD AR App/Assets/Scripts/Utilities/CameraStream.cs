using System;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.MagicLeap;
using static UnityEngine.XR.MagicLeap.MLCameraBase.Metadata;
using Debug = UnityEngine.Debug;
using MagicLeap.Android;
using UnityEngine.Android;

/// <summary>
/// Streams Magic Leap 2 camera video frames in RGBA format and exposes the latest frame
/// as Texture2D for Sentis inference and byte[] for CPU preprocessing.
/// </summary>
public class ImageStream : MonoBehaviour
{
    private const MLCamera.Identifier CameraIdentifier = MLCamera.Identifier.CV;

    [SerializeField, Tooltip("Optional centralized permission component. If assigned, this script waits for it instead of requesting camera permission itself.")]
    private PermissionRequester permissionRequester;

    [SerializeField, Tooltip("Fallback to local camera permission request when no permission requester is assigned.")]
    private bool requestPermissionWhenNoRequester = true;

    [SerializeField, Tooltip("The UI RawImage to display camera video")]
    private RawImage _screenRendererRGB = null;

    [SerializeField, Tooltip("Desired capture width")]
    private int _captureWidth = 1920;

    [SerializeField, Tooltip("Desired capture height")]
    private int _captureHeight = 1080;

    [SerializeField, Tooltip("Run AE/AWB once before starting video")]
    private bool _runAeAwbBeforeVideoStart = true;

    private bool isCameraConnected;
    private bool cameraDeviceAvailable;
    private bool isCapturingVideo;
    private MLCamera colorCamera;

    private Texture2D videoTexture;

    // Latest frame cache for CV/ML (e.g., Sentis FaceBlaze).
    private readonly object frameLock = new object();
    private byte[] latestFrameRgba;
    private int latestFrameWidth;
    private int latestFrameHeight;
    private bool hasNewFrame;

    // Projection cache captured from the same camera callback used for frame bytes.
    private readonly object projectionLock = new object();
    private MLCamera.IntrinsicCalibrationParameters latestIntrinsics;
    private Matrix4x4 latestFramePose = Matrix4x4.identity;
    private bool hasProjectionSample;
    private bool hasValidIntrinsics;
    private bool hasValidFramePose;
    private long latestFrameTimestamp;
    private bool startupTriggered;

    private void Awake()
    {
        if (permissionRequester == null)
        {
            permissionRequester = FindAnyObjectByType<PermissionRequester>();
        }

        if (permissionRequester != null)
        {
            permissionRequester.OnPermissionsResolved += OnPermissionsResolved;
            if (permissionRequester.IsCameraGranted)
            {
                StartCameraCaptureOnce();
            }
            Debug.Log("[ImageStream] Awake: waiting on PermissionRequester for camera permission.");
        }
        else if (requestPermissionWhenNoRequester)
        {
            Permissions.RequestPermissions(
                new[] { Permission.Camera },
                OnPermissionGranted,
                OnPermissionDenied,
                OnPermissionDenied);

            Debug.Log("[ImageStream] Awake: requested camera permission directly.");
        }
        else
        {
            Debug.LogWarning("[ImageStream] No permission requester assigned and direct permission fallback disabled.");
        }
    }

    private void OnDisable()
    {
        if (permissionRequester != null)
        {
            permissionRequester.OnPermissionsResolved -= OnPermissionsResolved;
        }

        DisableMLCamera();

        if (videoTexture != null)
        {
            Destroy(videoTexture);
            videoTexture = null;
        }
    }

    private void Update()
    {
        if (!hasNewFrame)
        {
            return;
        }

        byte[] frameBytes;
        int width;
        int height;

        lock (frameLock)
        {
            frameBytes = latestFrameRgba;
            width = latestFrameWidth;
            height = latestFrameHeight;
            hasNewFrame = false;
        }

        if (frameBytes == null || width <= 0 || height <= 0)
        {
            return;
        }

        UpdateRGBTexture(frameBytes, width, height);
    }

    private void OnPermissionDenied(string permission)
    {
        MLPluginLog.Error($"{permission} denied, camera stream unavailable.");
    }

    private void OnPermissionGranted(string permission)
    {
        StartCameraCaptureOnce();
    }

    private void OnPermissionsResolved(bool allGranted)
    {
        if (!allGranted || permissionRequester == null || !permissionRequester.IsCameraGranted)
        {
            MLPluginLog.Error("[ImageStream] Camera permission not granted through PermissionRequester.");
            return;
        }

        StartCameraCaptureOnce();
    }

    private void StartCameraCaptureOnce()
    {
        if (startupTriggered)
        {
            return;
        }

        startupTriggered = true;
        StartCoroutine(EnableMLCamera());
    }

    private IEnumerator EnableMLCamera()
    {
        while (!cameraDeviceAvailable)
        {
            MLResult result = MLCamera.GetDeviceAvailabilityStatus(CameraIdentifier, out cameraDeviceAvailable);
            if (!(result.IsOk && cameraDeviceAvailable))
            {
                yield return new WaitForSeconds(1.0f);
            }
        }

        ConnectCamera();

        while (!isCameraConnected)
        {
            yield return null;
        }

        ConfigureAndPrepareVideoCapture();
        StartVideoCapture();
    }

    private async void ConnectCamera()
    {
        MLCamera.ConnectContext context = MLCamera.ConnectContext.Create();
        context.CamId = CameraIdentifier;
        context.EnableVideoStabilization = true;
        context.Flags = MLCameraBase.ConnectFlag.CamOnly;

        Debug.Log($"[ImageStream] Connecting to camera stream: {CameraIdentifier}");

        colorCamera = await MLCamera.CreateAndConnectAsync(context);
        if (colorCamera != null)
        {
            colorCamera.OnRawVideoFrameAvailable += OnRawVideoFrameAvailable;
            isCameraConnected = true;
            Debug.Log("[ImageStream] Camera connected successfully.");
        }
        else
        {
            Debug.LogError("[ImageStream] Failed to connect to camera.");
        }
    }

    private void ConfigureAndPrepareVideoCapture()
    {
        MLCamera.CaptureStreamConfig[] streamConfigs = new MLCamera.CaptureStreamConfig[1]
        {
            new MLCamera.CaptureStreamConfig()
            {
                OutputFormat = MLCamera.OutputFormat.RGBA_8888,
                CaptureType = MLCamera.CaptureType.Video,
                Width = _captureWidth,
                Height = _captureHeight
            }
        };

        MLCamera.CaptureConfig captureConfig = new MLCamera.CaptureConfig()
        {
            StreamConfigs = streamConfigs,
            CaptureFrameRate = MLCamera.CaptureFrameRate._30FPS
        };

        MLResult prepareResult = colorCamera.PrepareCapture(captureConfig, out MLCamera.Metadata _);
        if (!prepareResult.IsOk)
        {
            Debug.LogError("[ImageStream] PrepareCapture failed for video stream.");
        }
    }

    private void StartVideoCapture()
    {
        if (colorCamera == null || !isCameraConnected)
        {
            return;
        }

        if (_runAeAwbBeforeVideoStart)
        {
            MLResult aeAwbResult = colorCamera.PreCaptureAEAWB();
            if (!aeAwbResult.IsOk)
            {
                Debug.LogWarning("[ImageStream] PreCaptureAEAWB failed before video start.");
            }
        }

        MLResult startResult = colorCamera.CaptureVideoStart();
        isCapturingVideo = startResult.IsOk;

        if (!isCapturingVideo)
        {
            Debug.LogError("[ImageStream] CaptureVideoStart failed.");
        }
    }

    private void DisableMLCamera()
    {
        if (colorCamera == null)
        {
            return;
        }

        colorCamera.OnRawVideoFrameAvailable -= OnRawVideoFrameAvailable;

        if (isCapturingVideo)
        {
            colorCamera.CaptureVideoStop();
            isCapturingVideo = false;
        }

        colorCamera.Disconnect();
        isCameraConnected = false;
    }

    private void OnRawVideoFrameAvailable(
        MLCamera.CameraOutput output,
        MLCamera.ResultExtras extras,
        MLCamera.Metadata metadataHandle)
    {
        if (output.Format != MLCameraBase.OutputFormat.RGBA_8888)
            return;

        // Fix upside-down image.
        MLCamera.FlipFrameVertically(ref output);

        CacheLatestProjectionData(extras);

        // Keep a packed RGBA frame cache that can be reused by RawImage and Sentis preprocessing.
        CacheLatestFrame(output.Planes[0]);
    }

    private void CacheLatestProjectionData(MLCamera.ResultExtras extras)
    {
        bool intrinsicsValid = extras.Intrinsics.HasValue;
        bool poseValid = MLCVCamera.GetFramePose(extras.VCamTimestamp, out Matrix4x4 framePose).IsOk;
        long timestamp = Convert.ToInt64(extras.VCamTimestamp);

        lock (projectionLock)
        {
            hasProjectionSample = true;
            hasValidIntrinsics = intrinsicsValid;
            hasValidFramePose = poseValid;
            latestFrameTimestamp = timestamp;

            if (intrinsicsValid)
            {
                latestIntrinsics = extras.Intrinsics.Value;
            }

            if (poseValid)
            {
                latestFramePose = framePose;
            }
        }
    }

    private void CacheLatestFrame(MLCamera.PlaneInfo imagePlane)
    {
        int width = (int)imagePlane.Width;
        int height = (int)imagePlane.Height;
        int actualWidth = (int)(imagePlane.Width * imagePlane.PixelStride);
        int packedLength = actualWidth * height;

        lock (frameLock)
        {
            if (latestFrameRgba == null || latestFrameRgba.Length != packedLength)
            {
                latestFrameRgba = new byte[packedLength];
            }

            if (imagePlane.Stride != actualWidth)
            {
                for (int row = 0; row < height; row++)
                {
                    Buffer.BlockCopy(imagePlane.Data, row * (int)imagePlane.Stride, latestFrameRgba, row * actualWidth, actualWidth);
                }
            }
            else
            {
                Buffer.BlockCopy(imagePlane.Data, 0, latestFrameRgba, 0, packedLength);
            }

            latestFrameWidth = width;
            latestFrameHeight = height;
            hasNewFrame = true;
        }
    }

    private void UpdateRGBTexture(byte[] frameBytes, int width, int height)
    {
        if (videoTexture == null || videoTexture.width != width || videoTexture.height != height)
        {
            if (videoTexture != null)
                Destroy(videoTexture);

            videoTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
            videoTexture.filterMode = FilterMode.Bilinear;

            if (_screenRendererRGB != null)
                _screenRendererRGB.texture = videoTexture;
        }

        videoTexture.LoadRawTextureData(frameBytes);
        videoTexture.Apply(false);
    }

    /// <summary>
    /// Sentis-ready path: use this texture as model input source when possible.
    /// </summary>
    public Texture2D GetLatestFrameTexture()
    {
        return videoTexture;
    }

    /// <summary>
    /// CPU path for custom preprocessing. Returns a copy to avoid mutation races.
    /// </summary>
    public bool TryGetLatestFrameRgba(out byte[] frame, out int width, out int height)
    {
        lock (frameLock)
        {
            if (latestFrameRgba == null || latestFrameWidth <= 0 || latestFrameHeight <= 0)
            {
                frame = null;
                width = 0;
                height = 0;
                return false;
            }

            frame = new byte[latestFrameRgba.Length];
            Buffer.BlockCopy(latestFrameRgba, 0, frame, 0, latestFrameRgba.Length);
            width = latestFrameWidth;
            height = latestFrameHeight;
            return true;
        }
    }

    /// <summary>
    /// Returns the latest projection data captured with the camera frame callback.
    /// </summary>
    public bool TryGetLatestProjectionData(
        out MLCamera.IntrinsicCalibrationParameters intrinsics,
        out Matrix4x4 framePose,
        out long timestamp,
        out bool intrinsicsValid,
        out bool poseValid)
    {
        lock (projectionLock)
        {
            intrinsics = latestIntrinsics;
            framePose = latestFramePose;
            timestamp = latestFrameTimestamp;
            intrinsicsValid = hasValidIntrinsics;
            poseValid = hasValidFramePose;

            return hasProjectionSample;
        }
    }
}