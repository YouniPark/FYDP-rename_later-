using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.MagicLeap;
using InputDevice = UnityEngine.XR.InputDevice;

public class GazeAndCameraPermissionManager : MonoBehaviour
{
    private InputDevice _eyeTrackingDevice;

    public bool EyeTrackingPermissionGranted { get; private set; }
    public bool CameraPermissionGranted { get; private set; }

    public Vector3 GazePosition { get; private set; }
    public Quaternion GazeRotation { get; private set; }

    public bool HasGazePose { get; private set; }

    public Vector3 GazeOriginWorld => GazePosition;
    public Vector3 GazeDirectionWorld => (GazeRotation * Vector3.forward).normalized;

    private void Start()
    {
        // Request Eye Tracking
        MagicLeap.Android.Permissions.RequestPermission(
            MagicLeap.Android.Permissions.EyeTracking,
            OnEyeTrackingGranted,
            OnEyeTrackingDenied,
            OnEyeTrackingDenied
        );

        // Request Camera
        MagicLeap.Android.Permissions.RequestPermission(
            UnityEngine.Android.Permission.Camera,
            OnCameraGranted,
            OnCameraDenied,
            OnCameraDenied
        );
    }

    private static readonly InputFeatureUsage<Vector3> GazePositionUsage = new InputFeatureUsage<Vector3>("GazePosition");
    private static readonly InputFeatureUsage<Quaternion> GazeRotationUsage = new InputFeatureUsage<Quaternion>("GazeRotation");

    private void Update()
    {

        if (!EyeTrackingPermissionGranted)
        {
            HasGazePose = false;
            return;
        }

        if (!_eyeTrackingDevice.isValid)
        {
            var list = new List<InputDevice>();
            InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.EyeTracking, list);
            if (list.Count > 0)
                _eyeTrackingDevice = list[0];

            if (!_eyeTrackingDevice.isValid)
            {
                Debug.LogWarning("[GazePerms] Eye tracking device not available yet.");
                HasGazePose = false;
                return;
            }
        }

        bool ok = _eyeTrackingDevice.TryGetFeatureValue(CommonUsages.isTracked, out bool isTracked);
        ok &= _eyeTrackingDevice.TryGetFeatureValue(GazePositionUsage, out Vector3 pos);
        ok &= _eyeTrackingDevice.TryGetFeatureValue(GazeRotationUsage, out Quaternion rot);

        if (ok && isTracked)
        {
            GazePosition = pos;
            GazeRotation = rot;
            HasGazePose = true;
        }
        else
        {
            HasGazePose = false;
        }
    }

    private void OnEyeTrackingDenied(string permission)
    {
        EyeTrackingPermissionGranted = false;
        Debug.LogError("[GazePerms] Eye tracking permission denied.");
    }

    private void OnEyeTrackingGranted(string permission)
    {
        EyeTrackingPermissionGranted = true;
        Debug.Log("[GazePerms] Eye tracking permission granted.");
    }

    private void OnCameraDenied(string permission)
    {
        CameraPermissionGranted = false;
        Debug.LogError("[GazePerms] Camera permission denied.");
    }

    private void OnCameraGranted(string permission)
    {
        CameraPermissionGranted = true;
        Debug.Log("[GazePerms] Camera permission granted.");
    }
}
