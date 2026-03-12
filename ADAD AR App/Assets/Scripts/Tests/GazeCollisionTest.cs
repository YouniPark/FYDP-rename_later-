using MagicLeap.Android;
using MagicLeap.OpenXR.Features.EyeTracker;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.OpenXR;
using MagicLeap.Examples;
using System.Text;
using UnityEngine.UI;
using UnityEngine.XR.OpenXR.Features.Interactions;
using UnityEngine.XR.OpenXR.NativeTypes;

public class GazeCollisionTest : MonoBehaviour
{
    // ===== Inspector Settings =====
    
    /// <summary>
    /// Reference to the sphere renderer that will change color based on gaze
    /// </summary>
    [SerializeField] 
    private MeshRenderer sphereRenderer;
    
    /// <summary>
    /// Material for when the sphere is NOT being looked at
    /// </summary>
    [SerializeField] 
    private Material defaultMaterial;
    
    /// <summary>
    /// Material for when the sphere IS being looked at (gaze ray hits it)
    /// </summary>
    [SerializeField] 
    private Material highlightMaterial;

    /// <summary>
    /// When true, sphere highlight only activates after first fixation event in current hit window.
    /// When false, sphere highlights immediately on collision.
    /// </summary>
    [SerializeField]
    private bool highlightOnFixation = true;

    /// <summary>
    /// Minimum distance the gaze point must move before we update the fixation target
    /// Prevents jitter from small tracking noise
    /// </summary>
    [SerializeField] 
    private float movementThreshold = 0.01f;

    /// <summary>
    /// The point in 3D space where the user is looking (updated by gaze tracker)
    /// </summary>
    [SerializeField] 
    private Transform fixationPointTransform;

    // ===== Private Tracking State =====

    /// <summary>
    /// Cached reference to the main camera for raycasting
    /// </summary>
    private Camera mainCamera;

    /// <summary>
    /// The Magic Leap eye tracker feature that provides gaze data
    /// </summary>
    private MagicLeapEyeTrackerFeature eyeTrackerFeature;

    /// <summary>
    /// Latest eye tracker data (position, rotation, pupil size, etc.)
    /// </summary>
    private EyeTrackerData eyeTrackerData;

    /// <summary>
    /// List of available input devices (used to find the eye tracking device)
    /// </summary>
    private List<InputDevice> inputDeviceList = new();

    /// <summary>
    /// The XR input device that provides eye tracking data
    /// </summary>
    private InputDevice eyeTrackingDevice;

    // ===== Permission State =====

    private bool eyeTrackPermission = false;
    private bool pupilSizePermission = false;

    // ===== Initialization State =====

    private bool isInitialized = false;
    private bool isDeviceVerified = false;

    // ===== Gaze Type Selection =====
    private enum GazeType { EyeGazeExt, EyeGazeML, Fixation }
    [SerializeField] 
    private GazeType currentGazeType = GazeType.Fixation;

    // ===== Event Tracking =====
    
    /// <summary>
    /// Whether the sphere was hit by the gaze ray in the previous frame.
    /// Used to detect state changes for interaction start/end logging.
    /// </summary>
    private bool wasHitLastFrame = false;

    // ===== Gaze Behavior Tracking =====

    /// <summary>
    /// The most recently observed gaze behavior type while the sphere is being hit.
    /// Logged on change so device logs reveal the actual enum value names.
    /// </summary>
    private string lastBehaviorType;

    /// <summary>
    /// Prevents the FIXATION EVENT log from firing more than once per continuous gaze hit.
    /// Reset when the gaze ray leaves the sphere.
    /// </summary>
    private bool fixationLoggedForCurrentHit;

    // Called when the script instance is being loaded 
    private void Awake()
    {
        // Request the permissions needed for eye tracking
        Permissions.RequestPermissions(
            new string[] { Permissions.EyeTracking, Permissions.PupilSize }, 
            OnPermissionGranted, 
            OnPermissionDenied, 
            OnPermissionDenied
        );

        // Cache the main camera for raycasting
        mainCamera = Camera.main;

        Debug.Log("[GazeCollisionTest] Awake: Permissions requested, camera cached.");
    }

    // Called every frame
    private void Update()
    {
        // Check if we have the required permissions
        if (!ArePermissionsGranted())
            return;

        // Initialize the eye tracker feature if not already done
        if (!isInitialized)
        {
            Initialize();
            return;
        }

        // Verify the eye tracking device is valid and has data
        if (IsEyeTrackingDeviceValid())
        {
            // Perform gaze raycasting and update sphere material
            ShowEyeTrackingVisualization();
        }
        else    // Device is not valid
        {
            // Reset sphere to default
            if (sphereRenderer != null)
                sphereRenderer.sharedMaterial = defaultMaterial;
        }
    }

    /// <summary>
    /// Called when a permission is successfully granted
    /// </summary>
    private void OnPermissionGranted(string permission)
    {
        if (permission == Permissions.EyeTracking)
        {
            eyeTrackPermission = true;
            Debug.Log("[GazeCollisionTest] EyeTracking permission granted.");
        }

        if (permission == Permissions.PupilSize)
        {
            pupilSizePermission = true;
            Debug.Log("[GazeCollisionTest] PupilSize permission granted.");
        }
    }

    /// <summary>
    /// Called when a permission request is denied or fails
    /// </summary>
    private void OnPermissionDenied(string permission)
    {
        Debug.LogError($"[GazeCollisionTest] Permission denied: {permission}. Gaze tracking will not work.");
    }

    /// <summary>
    /// Check if both required permissions have been granted
    /// </summary>
    private bool ArePermissionsGranted()
    {
        return eyeTrackPermission && pupilSizePermission;
    }

    /// <summary>
    /// Initialize the Magic Leap eye tracker feature
    /// Called once on first valid Update after permissions are granted
    /// </summary>
    private void Initialize()
    {
        // Copied from GazeTrackingExample.cs
        eyeTrackerFeature = OpenXRSettings.Instance.GetFeature<MagicLeapEyeTrackerFeature>();
        eyeTrackerFeature.CreateEyeTracker();
        isInitialized = true;

        Debug.Log("[GazeCollisionTest] Initialize: Eye tracker created and ready.");
    }

    /// <summary>
    /// Verify that the eye tracking device is valid and has current data
    /// Handles both EyeGazeExt (XR input device) and EyeGazeML (native API) paths
    /// </summary>
    private bool IsEyeTrackingDeviceValid()
    {
        // EyeGazeExt path: try to find the eye tracking device from XR input system
        if (currentGazeType == GazeType.EyeGazeExt && (!eyeTrackingDevice.isValid || !isDeviceVerified))
        {
            // Query for devices with eye tracking characteristics
            InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.EyeTracking, inputDeviceList);

            // Find the specific eye tracking device by name
            eyeTrackingDevice = inputDeviceList.Find(device => device.name == "Eye Tracking OpenXR");

            if (eyeTrackingDevice == null || !eyeTrackingDevice.isValid)
            {
                Debug.LogWarning("[GazeCollisionTest] Could not find valid eye tracking device.");
                return false;
            }
        }
        // EyeGazeML and Fixation paths: use native API
        else if (eyeTrackingDevice.isValid || currentGazeType != GazeType.EyeGazeExt)
        {
            // Get the latest eye tracker data from the feature
            eyeTrackerData = eyeTrackerFeature.GetEyeTrackerData();

            // Check if the data is valid
            if (eyeTrackerData.PosesData.Result != XrResult.Success)
            {
                Debug.LogWarning("[GazeCollisionTest] Eye tracker data invalid or unavailable.");
                return false;
            }
        }

        isDeviceVerified = true;
        return true;
    }

    /// <summary>
    /// Main visualization logic:
    /// 1. Get the current gaze position and rotation
    /// 2. Update the fixation point transform
    /// 3. Raycast from camera to fixation point
    /// 4. Update sphere material based on whether ray hits it
    /// </summary>
    private void ShowEyeTrackingVisualization()
    {
        // ===== Extract Gaze Data =====
        bool hasData = false;
        Vector3 gazePosition = Vector3.zero;
        Quaternion gazeRotation = Quaternion.identity;
        Vector3 offsetFromFace = Vector3.zero;

        // Different gaze sources provide data differently
        if (currentGazeType == GazeType.EyeGazeExt)
        {
            // EyeGazeExt: read from XR input device features
            hasData = eyeTrackingDevice.TryGetFeatureValue(CommonUsages.isTracked, out bool isTracked) & 
                      eyeTrackingDevice.TryGetFeatureValue(EyeTrackingUsages.gazePosition, out gazePosition) &
                      eyeTrackingDevice.TryGetFeatureValue(EyeTrackingUsages.gazeRotation, out gazeRotation);

            offsetFromFace = gazeRotation * Vector3.forward;
        }
        else if (currentGazeType == GazeType.EyeGazeML)
        {
            // EyeGazeML: use native Magic Leap eye tracker API for current gaze
            hasData = eyeTrackerData.PosesData.Result == XrResult.Success;
            gazePosition = eyeTrackerData.PosesData.GazePose.Pose.position;
            gazeRotation = eyeTrackerData.PosesData.GazePose.Pose.rotation;
            offsetFromFace = gazeRotation * Vector3.forward;
        }
        else if (currentGazeType == GazeType.Fixation)
        {
            // Fixation: use the stabilized fixation point instead of raw gaze
            hasData = eyeTrackerData.PosesData.Result == XrResult.Success;
            gazePosition = eyeTrackerData.PosesData.FixationPose.Pose.position;
            gazeRotation = eyeTrackerData.PosesData.FixationPose.Pose.rotation;
            offsetFromFace = Vector3.zero; // Fixation is already a point, no offset needed
        }

        if (!hasData)
            return;

        // ===== Update Fixation Point Transform =====
        Vector3 targetGazePosition = gazePosition + offsetFromFace;

        // Only update if movement is large enough (avoid jitter)
        if (Vector3.Distance(fixationPointTransform.position, targetGazePosition) >= movementThreshold)
        {
            fixationPointTransform.SetLocalPositionAndRotation(targetGazePosition, gazeRotation);
        }

        // ===== Raycast from Camera to Fixation Point =====
        Vector3 rayDirection = fixationPointTransform.position - mainCamera.transform.position;
        var ray = new Ray(mainCamera.transform.position, rayDirection);

        // ===== Update Sphere Material Based on Raycast Result =====
        bool isSphereHit = false;

        if (Physics.Raycast(ray, out RaycastHit hitInfo))
        {
            // A collider was hit by the ray
            if (hitInfo.transform.gameObject == sphereRenderer.gameObject)
            {
                // The hit object is our sphere
                isSphereHit = true;
            }
            else
            {
                // Ray hit something else, not our sphere
                isSphereHit = false;
            }
        }
        else
        {
            // Ray hit nothing
            isSphereHit = false;
        }

        // ===== Track Gaze Behavior =====
        // While the sphere is being hit, poll the gaze behavior type each frame.
        if (isSphereHit)
        {
            GazeBehavior gazeBehavior = eyeTrackerData.GazeBehaviorData;
            string behaviorName = gazeBehavior.GazeBehaviorType.ToString();

            if (behaviorName != lastBehaviorType)
            {
                Debug.Log($"[GazeCollisionTest] Gaze behavior: {behaviorName}");
                lastBehaviorType = behaviorName;
            }

            // First fixation event on this continuous hit — fires once per gaze engagement
            if (!fixationLoggedForCurrentHit && behaviorName.ToLower().Contains("fixation"))
            {
                Debug.Log("[GazeCollisionTest] FIXATION EVENT: First fixation detected while looking at the sphere.");
                fixationLoggedForCurrentHit = true;
            }
        }
        else if (!isSphereHit && wasHitLastFrame)
        {
            // Gaze left sphere — reset behavior tracking for the next hit
            lastBehaviorType = null;
            fixationLoggedForCurrentHit = false;
        }

        // ===== Apply Highlight =====
        bool shouldHighlight = isSphereHit && (!highlightOnFixation || fixationLoggedForCurrentHit);
        sphereRenderer.sharedMaterial = shouldHighlight ? highlightMaterial : defaultMaterial;

        // ===== Log State Changes =====
        // Log when the interaction state changes (hit -> no hit or no hit -> hit)
        if (isSphereHit && !wasHitLastFrame)
        {
            Debug.Log("[GazeCollisionTest] INTERACTION START: Gaze ray is now hitting the sphere!");
        }
        else if (!isSphereHit && wasHitLastFrame)
        {
            Debug.Log("[GazeCollisionTest] INTERACTION END: Gaze ray left the sphere.");
        }

        wasHitLastFrame = isSphereHit;
    }
}
