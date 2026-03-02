using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

public class MLCameraFrameProvider : MonoBehaviour
{
    [Serializable]
    public struct CameraIntrinsics
    {
        public bool IsValid;
        public float Fx;
        public float Fy;
        public float Cx;
        public float Cy;
    }

    public struct CameraFrame
    {
        public Texture Texture;
        public Color32[] Pixels;
        public int Width;
        public int Height;
        public long TimestampNs;
        public Pose CameraPoseWorld;
        public CameraIntrinsics Intrinsics;
    }

    [Header("Fallback webcam settings")]
    public int requestedWidth = 1280;
    public int requestedHeight = 720;
    public int requestedFps = 30;

    [Header("Optional: assign if camera origin differs")]
    public Transform cameraOpticalTransform;

    public bool IsRunning { get; private set; }
    public Texture CurrentTexture => _texture;

    private WebCamTexture _webCam;
    private Texture _texture;
    private Color32[] _pixelBuffer;
    private readonly Queue<CameraFrame> _frames = new Queue<CameraFrame>(4);

    // Reflection handles for Magic Leap MLCamera path.
    private object _mlCamera;
    private MethodInfo _mlDisconnect;
    private bool _usingMLCamera;
    public GazeAndCameraPermissionManager perms;

    private void Start()
    {
        StartProvider();
    }

    private void OnDestroy()
    {
        StopProvider();
    }

    public void StartProvider()
    {
        if (IsRunning) return;
        if (!EnsurePermissions()) return;

        _usingMLCamera = TryStartMLCamera();
        if (!_usingMLCamera)
        {
            StartFallbackWebcam();
        }

        IsRunning = _texture != null;
    }

    public void StopProvider()
    {
        if (_webCam != null)
        {
            _webCam.Stop();
            _webCam = null;
        }

        if (_mlCamera != null && _mlDisconnect != null)
        {
            _mlDisconnect.Invoke(_mlCamera, null);
            _mlCamera = null;
        }

        IsRunning = false;
    }

    private void Update()
    {
        if (!IsRunning || _texture == null) return;

        int width;
        int height;

        if (_webCam != null)
        {
            if (!_webCam.didUpdateThisFrame) return;
            width = _webCam.width;
            height = _webCam.height;
            if (_pixelBuffer == null || _pixelBuffer.Length != width * height)
            {
                _pixelBuffer = new Color32[width * height];
            }
            _webCam.GetPixels32(_pixelBuffer);
        }
        else
        {
            width = _texture.width;
            height = _texture.height;
            _pixelBuffer = null;
        }

        var intrinsics = BuildApproxIntrinsics(width, height);
        var pose = cameraOpticalTransform != null
            ? new Pose(cameraOpticalTransform.position, cameraOpticalTransform.rotation)
            : new Pose(transform.position, transform.rotation);

        var frame = new CameraFrame
        {
            Texture = _texture,
            Pixels = _pixelBuffer,
            Width = width,
            Height = height,
            TimestampNs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000L,
            CameraPoseWorld = pose,
            Intrinsics = intrinsics
        };

        if (_frames.Count > 2) _frames.Dequeue();
        _frames.Enqueue(frame);
    }

    public bool TryGetLatestFrame(out CameraFrame frame)
    {
        if (_frames.Count > 0)
        {
            frame = _frames.Peek();
            return true;
        }

        frame = default;
        return false;
    }

    private CameraIntrinsics BuildApproxIntrinsics(int width, int height)
    {
        float fx = 0.5f * width;
        float fy = 0.5f * height;
        float cx = width * 0.5f;
        float cy = height * 0.5f;

        return new CameraIntrinsics
        {
            IsValid = false,
            Fx = fx,
            Fy = fy,
            Cx = cx,
            Cy = cy
        };
    }

        private bool EnsurePermissions()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
            Debug.LogWarning("Camera permission requested. Start will continue after user grants permission.");
            return false;
        }
#endif
        return true;
    }

    private void StartFallbackWebcam()
    {
        _webCam = new WebCamTexture(requestedWidth, requestedHeight, requestedFps);
        _webCam.Play();
        _texture = _webCam;
        Debug.LogWarning("MLCamera not found; using WebCamTexture fallback.");
    }

    private bool TryStartMLCamera()
    {
#if UNITY_ANDROID
        try
        {
            var mlCameraType = Type.GetType("UnityEngine.XR.MagicLeap.MLCamera, Unity.XR.MagicLeap")
                               ?? Type.GetType("UnityEngine.XR.MagicLeap.MLCamera, MagicLeap")
                               ?? Type.GetType("UnityEngine.XR.MagicLeap.MLCamera, Assembly-CSharp");

            if (mlCameraType == null) return false;

            var createAndConnect = mlCameraType.GetMethod("CreateAndConnect", BindingFlags.Public | BindingFlags.Static);
            _mlDisconnect = mlCameraType.GetMethod("Disconnect", BindingFlags.Public | BindingFlags.Instance);
            if (createAndConnect == null || _mlDisconnect == null) return false;

            _mlCamera = createAndConnect.GetParameters().Length == 0
                ? createAndConnect.Invoke(null, null)
                : null;

            if (_mlCamera != null)
            {
                Debug.Log("MLCamera connected (reflection path). Configure raw frame callbacks per ML plugin version.");
                return false; 
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"MLCamera reflection path unavailable: {ex.Message}");
        }
#endif
        return false;
    }
}
