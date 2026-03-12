using System;
using System.Collections.Generic;
using MagicLeap.Android;
using UnityEngine;
using UnityEngine.Android;

/// <summary>
/// Requests all runtime permissions required by the integrated face+gaze system.
/// </summary>
public class PermissionRequester : MonoBehaviour
{
    [SerializeField] private bool requestOnAwake = true;

    public bool IsCameraGranted { get; private set; }
    public bool IsEyeTrackingGranted { get; private set; }
    public bool IsPupilSizeGranted { get; private set; }

    public bool AreAllGranted => IsCameraGranted && IsEyeTrackingGranted && IsPupilSizeGranted;

    public event Action<bool> OnPermissionsResolved;

    private bool _hasRequested;
    private bool _hasResolved;
    private readonly HashSet<string> _resolvedPermissions = new HashSet<string>();
    private const int RequiredPermissionCount = 3;

    private void Awake()
    {
        if (requestOnAwake)
        {
            RequestAllPermissions();
        }
    }

    public void RequestAllPermissions()
    {
        if (_hasRequested)
        {
            return;
        }

        _hasRequested = true;
        Permissions.RequestPermissions(
            new[] { Permission.Camera, Permissions.EyeTracking, Permissions.PupilSize },
            OnPermissionGranted,
            OnPermissionDenied,
            OnPermissionDenied);
    }

    private void OnPermissionGranted(string permission)
    {
        _resolvedPermissions.Add(permission);

        if (permission == Permission.Camera)
        {
            IsCameraGranted = true;
        }
        else if (permission == Permissions.EyeTracking)
        {
            IsEyeTrackingGranted = true;
        }
        else if (permission == Permissions.PupilSize)
        {
            IsPupilSizeGranted = true;
        }

        TryResolve();
    }

    private void OnPermissionDenied(string permission)
    {
        _resolvedPermissions.Add(permission);
        Debug.LogError($"[PermissionRequester] Permission denied: {permission}");
        TryResolve();
    }

    private void TryResolve()
    {
        if (_hasResolved)
        {
            return;
        }

        if (_resolvedPermissions.Count < RequiredPermissionCount)
        {
            return;
        }

        if (AreAllGranted)
        {
            _hasResolved = true;
            Debug.Log("[PermissionRequester] All required permissions granted.");
            OnPermissionsResolved?.Invoke(true);
            return;
        }

        _hasResolved = true;
        OnPermissionsResolved?.Invoke(false);
    }
}
