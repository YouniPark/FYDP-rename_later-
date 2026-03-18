using UnityEngine;

/// <summary>
/// Lightweight coordinator for the integrated face-detection + gaze-interaction scene.
///
/// Scene setup:
///   Assign all references in this component's Inspector, or leave them null to auto-wire
///   via scene search. Prefer explicit assignment in production for clarity.
///
/// Data flow:
///   PermissionRequester → grants Camera + EyeTracking + PupilSize
///   ImageStream         → streams Magic Leap CV camera and exposes frames + intrinsics
///   BlazeFaceDetector   → runs inference on frames, fires OnFacesDetected events
///   FaceProxyProjector  → projects detected faces into world-space proxy spheres
///   EyeGazeRayProvider  → wraps ML eye tracker, exposes gaze ray + behavior each frame
///   FaceProxyGazeInteractor → casts gaze ray against proxy colliders, updates hover state
/// </summary>
public class FaceGazeSystemMain : MonoBehaviour
{
    [Header("System Components")]
    [SerializeField] private PermissionRequester permissionRequester;
    [SerializeField] private ImageStream cameraProvider;
    [SerializeField] private BlazeFaceDetector faceDetector;
    [SerializeField] private FaceProxyProjector faceProxyProjector;
    [SerializeField] private EyeGazeRayProvider gazeProvider;
    [SerializeField] private FaceProxyGazeInteractor gazeInteractor;

    private void Awake()
    {
        // Fill in any references that weren't assigned in the Inspector
        AutoWireIfMissing();

        // Push the resolved provider into the interactor so both ends know about each other
        if (gazeInteractor != null && gazeProvider != null)
        {
            gazeInteractor.SetGazeProvider(gazeProvider);
        }

        if (permissionRequester == null)
        {
            Debug.LogWarning("[FaceGazeSystemMain] PermissionRequester is not assigned.");
        }

        if (cameraProvider == null || faceDetector == null || faceProxyProjector == null || gazeProvider == null || gazeInteractor == null)
        {
            Debug.LogWarning("[FaceGazeSystemMain] One or more required system references are missing.");
            return;
        }

        Debug.Log("[FaceGazeSystemMain] Full system wiring validated.");
    }

    private void AutoWireIfMissing()
    {
        if (permissionRequester == null)
        {
            permissionRequester = FindFirstObjectByType<PermissionRequester>();
        }

        if (cameraProvider == null)
        {
            cameraProvider = FindFirstObjectByType<ImageStream>();
        }

        if (faceDetector == null)
        {
            faceDetector = FindFirstObjectByType<BlazeFaceDetector>();
        }

        if (faceProxyProjector == null)
        {
            faceProxyProjector = FindFirstObjectByType<FaceProxyProjector>();
        }

        if (gazeProvider == null)
        {
            gazeProvider = FindFirstObjectByType<EyeGazeRayProvider>();
        }

        if (gazeInteractor == null)
        {
            gazeInteractor = FindFirstObjectByType<FaceProxyGazeInteractor>();
        }
    }
}
