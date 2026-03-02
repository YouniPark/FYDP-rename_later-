using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.OpenXR;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

#if UNITY_ANDROID
using MagicLeap.OpenXR.Features.EyeTracker;
#endif

public class EyeTrackerProvider : MonoBehaviour
{
    public struct GazeSample
    {
        public bool IsValid;
        public bool IsFixatingEvent;
        public bool IsCalibrated;
        public long TimestampNs;
        public Vector3 OriginWorld;
        public Vector3 DirectionWorld;
    }

    public GazeSample LatestSample { get; private set; }

    private InputDevice _eyeDevice;
    private readonly List<InputDevice> _devices = new List<InputDevice>();
    private bool _prevGazeValid;

    private void Start()
    {
        if (!EnsurePermissions()) return;
        ValidateFeatureEnabled();
        TryBindEyeDevice();
    }

    private void Update()
    {
        if (!_eyeDevice.isValid)
            TryBindEyeDevice();

        var sample = new GazeSample
        {
            IsValid = false,
            IsFixatingEvent = false,
            IsCalibrated = true,
            TimestampNs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1_000_000L
        };

        if (_eyeDevice.isValid && _eyeDevice.TryGetFeatureValue(CommonUsages.eyesData, out Eyes eyes))
        {
            if (eyes.TryGetFixationPoint(out var fixationPoint))
            {
                var centerEye = Camera.main != null ? Camera.main.transform.position : Vector3.zero;
                var dir = (fixationPoint - centerEye).normalized;

                sample.IsValid = dir.sqrMagnitude > 0.01f;
                sample.OriginWorld = centerEye;
                sample.DirectionWorld = dir;
            }

            if (eyes.TryGetLeftEyeOpenAmount(out var leftOpen) && eyes.TryGetRightEyeOpenAmount(out var rightOpen))
            {
                sample.IsCalibrated = !(Mathf.Approximately(leftOpen, 0f) && Mathf.Approximately(rightOpen, 0f));
            }
        }

        sample.IsFixatingEvent = sample.IsValid && !_prevGazeValid;
        _prevGazeValid = sample.IsValid;

        LatestSample = sample;
    }

    private const string EyeTrackingPermission = "com.magicleap.permission.EYE_TRACKING";

    private bool EnsurePermissions()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!Permission.HasUserAuthorizedPermission(EyeTrackingPermission))
        {
            Permission.RequestUserPermission(EyeTrackingPermission);
            Debug.LogWarning("Eye tracking permission requested.");
            return false;
        }
#endif
        return true;
    }

    private void ValidateFeatureEnabled()
    {
        if (OpenXRSettings.Instance == null)
        {
            Debug.LogError("EyeTrackerProvider: OpenXRSettings.Instance is null.");
            return;
        }

        bool found = false;
        foreach (var f in OpenXRSettings.Instance.GetFeatures())
        {
            if (f == null) continue;
            var name = f.GetType().FullName;
            if (!string.IsNullOrEmpty(name) && name.Contains("MagicLeap.OpenXR.Features.EyeTracker"))
            {
                found = f.enabled;
                break;
            }
        }

        if (!found)
        {
            Debug.LogError("EyeTrackerProvider: Magic Leap OpenXR Eye Tracker Feature is not enabled.");
        }
    }

    private void TryBindEyeDevice()
    {
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.EyeTracking, _devices);
        _eyeDevice = _devices.Count > 0 ? _devices[0] : default;

        if (!_eyeDevice.isValid)
        {
            InputDevices.GetDevices(_devices);
            foreach (var d in _devices)
            {
                if (d.TryGetFeatureValue(CommonUsages.eyesData, out _))
                {
                    _eyeDevice = d;
                    break;
                }
            }
        }
    }
}
