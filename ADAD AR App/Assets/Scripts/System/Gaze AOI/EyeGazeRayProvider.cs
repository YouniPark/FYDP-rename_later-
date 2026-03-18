using System.Collections.Generic;
using MagicLeap.OpenXR.Features.EyeTracker;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.OpenXR;
using UnityEngine.XR.OpenXR.Features.Interactions;
using UnityEngine.XR.OpenXR.NativeTypes;

/// <summary>
/// Provides a world-space gaze ray from Magic Leap eye tracking.
/// Other system components (e.g. FaceProxyGazeInteractor) call TryGetGazeRay() each frame.
/// Handles permission gating, tracker initialization, and all three gaze modes:
///   EyeGazeExt  – raw XR input device  
///   EyeGazeML   – native ML eye tracker API
///   Fixation    – stabilized fixation point (recommended for interaction)
/// </summary>
public class EyeGazeRayProvider : MonoBehaviour
{
    private enum GazeType
    {
        EyeGazeExt,
        EyeGazeML,
        Fixation
    }

    [Header("Dependencies")]
    [SerializeField] private PermissionRequester permissionRequester;

    [Header("Gaze Source")]
    [SerializeField] private GazeType currentGazeType = GazeType.Fixation;

    [Header("Fixation Point Tracking")]
    [SerializeField] private Transform fixationPointTransform;
    [SerializeField] private float movementThreshold = 0.01f;

    private readonly List<InputDevice> _inputDeviceList = new List<InputDevice>();

    private MagicLeapEyeTrackerFeature _eyeTrackerFeature;
    private EyeTrackerData _eyeTrackerData;
    private InputDevice _eyeTrackingDevice;
    private Camera _mainCamera;

    private bool _isInitialized;
    private bool _isDeviceVerified;
    private bool _eyePermissionReady;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    private void Awake()
    {
        _mainCamera = Camera.main;

        // Try to find a PermissionRequester in the scene if one wasn't dragged in via Inspector
        if (permissionRequester == null)
        {
            permissionRequester = FindFirstObjectByType<PermissionRequester>();
        }

        if (permissionRequester == null)
        {
            Debug.LogWarning("[EyeGazeRayProvider] PermissionRequester not assigned. Gaze will not activate until eye permissions are granted externally.");
            return;
        }

        permissionRequester.OnPermissionsResolved += OnPermissionsResolved;
        _eyePermissionReady = permissionRequester.IsEyeTrackingGranted && permissionRequester.IsPupilSizeGranted;
    }

    private void OnDestroy()
    {
        if (permissionRequester != null)
        {
            permissionRequester.OnPermissionsResolved -= OnPermissionsResolved;
        }
    }

    private void Update()
    {
        // Update fixation point transform each frame (same logic as GazeCollisionTest)
        if (!_eyePermissionReady || !_isInitialized || !_isDeviceVerified || fixationPointTransform == null)
        {
            return;
        }

        if (!IsEyeTrackingDeviceValid())
        {
            return;
        }

        if (!TryGetRawGazeData(out Vector3 gazePosition, out Quaternion gazeRotation, out Vector3 fixationPoint))
        {
            return;
        }

        // Calculate target position based on gaze type
        Vector3 targetPosition = fixationPoint;
        if (currentGazeType == GazeType.EyeGazeML || currentGazeType == GazeType.EyeGazeExt)
        {
            targetPosition = gazePosition + (gazeRotation * Vector3.forward);
        }

        // Only update if movement is large enough (avoid jitter)
        if (Vector3.Distance(fixationPointTransform.position, targetPosition) >= movementThreshold)
        {
            fixationPointTransform.SetLocalPositionAndRotation(targetPosition, gazeRotation);
        }
    }

    private void OnPermissionsResolved(bool allGranted)
    {
        _eyePermissionReady = permissionRequester != null &&
                              permissionRequester.IsEyeTrackingGranted &&
                              permissionRequester.IsPupilSizeGranted;

        if (!allGranted)
        {
            Debug.LogWarning("[EyeGazeRayProvider] Required eye permissions were not granted.");
        }
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /// <summary>
    /// Returns a world-space gaze ray this frame, or false if tracking is not ready.
    /// For Fixation mode the ray origin is Camera.main and direction points to the fixation point.
    /// For EyeGazeML/EyeGazeExt the ray originates at the gaze pose position.
    /// </summary>
    public bool TryGetGazeRay(out Ray gazeRay)
    {
        gazeRay = default;

        if (!_eyePermissionReady)
        {
            return false;
        }

        if (!_isInitialized)
        {
            Initialize();
            if (!_isInitialized)
            {
                return false;
            }
        }

        if (!IsEyeTrackingDeviceValid())
        {
            return false;
        }

        if (!TryGetRawGazeData(out Vector3 gazePosition, out Quaternion gazeRotation, out Vector3 fixationPoint))
        {
            return false;
        }

        if (currentGazeType == GazeType.Fixation)
        {
            Vector3 origin = _mainCamera != null ? _mainCamera.transform.position : gazePosition;
            Vector3 direction = fixationPoint - origin;
            if (direction.sqrMagnitude <= Mathf.Epsilon)
            {
                return false;
            }

            gazeRay = new Ray(origin, direction.normalized);
            return true;
        }

        gazeRay = new Ray(gazePosition, gazeRotation * Vector3.forward);
        return true;
    }

    /// <summary>
    /// Returns the current ML gaze behavior (Fixation, Saccade, Pursuit, etc.).
    /// Only valid for EyeGazeML and Fixation modes — returns false for EyeGazeExt.
    /// Call this while TryGetGazeRay() is returning true to get per-frame behavior data.
    /// The exact GazeBehaviorType enum names can be confirmed by logging on device.
    /// </summary>
    public bool TryGetGazeBehavior(out GazeBehavior behavior)
    {
        behavior = default;

        if (!_isInitialized || !_isDeviceVerified || currentGazeType == GazeType.EyeGazeExt)
        {
            return false;
        }

        if (_eyeTrackerData.PosesData.Result != XrResult.Success)
        {
            return false;
        }

        behavior = _eyeTrackerData.GazeBehaviorData;
        return true;
    }

    // -------------------------------------------------------------------------
    // Internals
    // -------------------------------------------------------------------------

    private void Initialize()
    {
        _eyeTrackerFeature = OpenXRSettings.Instance != null
            ? OpenXRSettings.Instance.GetFeature<MagicLeapEyeTrackerFeature>()
            : null;

        if (_eyeTrackerFeature == null)
        {
            Debug.LogWarning("[EyeGazeRayProvider] MagicLeapEyeTrackerFeature not available.");
            return;
        }

        _eyeTrackerFeature.CreateEyeTracker();
        _isInitialized = true;
        Debug.Log("[EyeGazeRayProvider] Eye tracker initialized.");
    }

    private bool IsEyeTrackingDeviceValid()
    {
        if (currentGazeType == GazeType.EyeGazeExt && (!_eyeTrackingDevice.isValid || !_isDeviceVerified))
        {
            InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.EyeTracking, _inputDeviceList);
            _eyeTrackingDevice = _inputDeviceList.Find(device => device.name == "Eye Tracking OpenXR");

            if (!_eyeTrackingDevice.isValid)
            {
                return false;
            }
        }
        else
        {
            _eyeTrackerData = _eyeTrackerFeature.GetEyeTrackerData();
            if (_eyeTrackerData.PosesData.Result != XrResult.Success)
            {
                return false;
            }
        }

        _isDeviceVerified = true;
        return true;
    }

    private bool TryGetRawGazeData(out Vector3 gazePosition, out Quaternion gazeRotation, out Vector3 fixationPoint)
    {
        gazePosition = Vector3.zero;
        gazeRotation = Quaternion.identity;
        fixationPoint = Vector3.zero;

        if (currentGazeType == GazeType.EyeGazeExt)
        {
            bool hasTracked = _eyeTrackingDevice.TryGetFeatureValue(CommonUsages.isTracked, out bool isTracked) && isTracked;
            bool hasPosition = _eyeTrackingDevice.TryGetFeatureValue(EyeTrackingUsages.gazePosition, out gazePosition);
            bool hasRotation = _eyeTrackingDevice.TryGetFeatureValue(EyeTrackingUsages.gazeRotation, out gazeRotation);
            fixationPoint = gazePosition + (gazeRotation * Vector3.forward);
            return hasTracked && hasPosition && hasRotation;
        }

        if (_eyeTrackerData.PosesData.Result != XrResult.Success)
        {
            return false;
        }

        if (currentGazeType == GazeType.EyeGazeML)
        {
            gazePosition = _eyeTrackerData.PosesData.GazePose.Pose.position;
            gazeRotation = _eyeTrackerData.PosesData.GazePose.Pose.rotation;
            fixationPoint = gazePosition + (gazeRotation * Vector3.forward);
            return true;
        }

        gazePosition = _eyeTrackerData.PosesData.FixationPose.Pose.position;
        gazeRotation = _eyeTrackerData.PosesData.FixationPose.Pose.rotation;
        fixationPoint = gazePosition;
        return true;
    }
}
