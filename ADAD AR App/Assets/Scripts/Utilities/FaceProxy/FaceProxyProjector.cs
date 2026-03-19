using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Projects 2D BlazeFace detections into world-space proxy objects.
/// </summary>
public class FaceProxyProjector : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private BlazeFaceDetector detector;
    [SerializeField] private ImageStream cameraSource;
    [SerializeField] private GameObject faceProxyPrefab;
    [SerializeField] private Transform proxyParent;
    [Tooltip("Assign the tracking-space root (usually XR Origin). If left null, frame pose is assumed world-space.")]
    [SerializeField] private Transform trackingSpaceTransform;
    [Tooltip("Optional explicit head/camera transform for pose debugging. If null, Camera.main is used.")]
    [SerializeField] private Transform debugHeadTransform;

    [Header("Proxy Visibility")]
    [Tooltip("Toggle to show/hide the face proxy objects. To be disabled with main version.")]
    [SerializeField] private bool visualizeFaceProxies = true;

    [Header("Proxy Limits")]
    [SerializeField, Min(1)] private int maxActiveProxies = 5;
    [SerializeField, Min(0f)] private float holdDuration = 0.35f;
    [SerializeField, Min(0f)] private float smoothingTime = 0.1f;

    [Header("Depth")]
    [SerializeField] private bool useAdaptiveDepth = true;
    [SerializeField, Min(0.05f)] private float minDepth = 0.4f;
    [SerializeField, Min(0.05f)] private float maxDepth = 1.3f;
    [SerializeField, Min(0.05f)] private float fixedDepthFallback = 1f;
    [Tooltip("Face area ratio treated as FAR (maps to maxDepth).")]
    [SerializeField, Min(0f)] private float faceAreaFar = 0.04f;
    [Tooltip("Face area ratio treated as NEAR (maps to minDepth).")]
    [SerializeField, Min(0f)] private float faceAreaNear = 0.16f;

    [Header("Debug Logging")]
    [SerializeField] private bool enableProjectorLogs = true;
    [SerializeField, Min(1)] private int missLogEveryNEvents = 20;
    [SerializeField] private bool logPoseOriginDelta = false;
    [SerializeField] private bool logFaceArea = false;
    [SerializeField, Min(1)] private int faceAreaLogEveryNDetections = 12;

    [Header("Ray Visualization")]
    [SerializeField] private bool visualizeRays = false;
    [Tooltip("Optional material for the debug ray LineRenderer. Use an unlit transparent material if you want alpha control.")]
    [SerializeField] private Material rayMaterial;
    [SerializeField, Min(0.001f)] private float rayWidth = 0.01f;

    [Header("Gaze Interaction")]
    [SerializeField] private bool addGazeTargetToSpawnedProxies = true;
    [SerializeField] private Material proxyDefaultMaterial;
    [SerializeField] private Material proxyHighlightMaterial;

    private sealed class ProxySlot
    {
        public Transform transform;
        public LineRenderer rayRenderer;
        public Renderer[] proxyRenderers;
        public Vector3 targetPosition;
        public Vector3 rayOrigin;
        public Vector3 velocity;
        public float lastSeenTime;
        public bool hasTarget;
    }

    private readonly List<ProxySlot> _slots = new List<ProxySlot>();
    private int _projectionDataMissCount;
    private int _missingReferenceCount;
    private int _lastLoggedFaceCount = -1;
    private int _detectionTickCount;
    private float _maxObservedFaceArea;
    private float _minObservedFaceArea = 1f;

    private Transform ResolvedHeadTransform => debugHeadTransform != null ? debugHeadTransform : (Camera.main != null ? Camera.main.transform : null);

    private void OnEnable()
    {
        if (detector != null)
        {
            detector.OnFacesDetected += OnFacesDetected;
            if (enableProjectorLogs)
            {
                Debug.Log("[FaceProxyProjector] Subscribed to BlazeFaceDetector.OnFacesDetected.");
            }
        }
        else if (enableProjectorLogs)
        {
            Debug.LogWarning("[FaceProxyProjector] Detector reference is missing.");
        }
    }

    private void OnDisable()
    {
        if (detector != null)
        {
            detector.OnFacesDetected -= OnFacesDetected;
        }

        for (int i = 0; i < _slots.Count; i++)
        {
            if (_slots[i].transform != null)
            {
                _slots[i].transform.gameObject.SetActive(false);
            }
        }
    }

    private void Update()
    {
        float now = Time.time;

        for (int i = 0; i < _slots.Count; i++)
        {
            ProxySlot slot = _slots[i];
            if (slot.transform == null)
            {
                continue;
            }

            bool withinHold = slot.hasTarget && (now - slot.lastSeenTime) <= holdDuration;
            if (!withinHold)
            {
                slot.transform.gameObject.SetActive(false);
                if (slot.rayRenderer != null)
                {
                    slot.rayRenderer.enabled = false;
                }
                continue;
            }

            if (!slot.transform.gameObject.activeSelf)
            {
                slot.transform.gameObject.SetActive(true);
            }

            SetRenderersEnabled(slot.proxyRenderers, visualizeFaceProxies);

            if (smoothingTime > 0f)
            {
                slot.transform.position = Vector3.SmoothDamp(
                    slot.transform.position,
                    slot.targetPosition,
                    ref slot.velocity,
                    smoothingTime,
                    Mathf.Infinity,
                    Time.deltaTime);
            }
            else
            {
                slot.transform.position = slot.targetPosition;
            }

            if (slot.rayRenderer != null)
            {
                ApplyRayRendererSettings(slot.rayRenderer);
                slot.rayRenderer.enabled = visualizeRays;
                if (visualizeRays)
                {
                    slot.rayRenderer.SetPosition(0, slot.rayOrigin);
                    slot.rayRenderer.SetPosition(1, slot.transform.position);
                }
            }

        }
    }

    private void OnFacesDetected(BlazeFaceDetector.DetectedFace[] faces)
    {
        if (cameraSource == null || faceProxyPrefab == null)
        {
            _missingReferenceCount++;
            if (enableProjectorLogs && (_missingReferenceCount == 1 || _missingReferenceCount % missLogEveryNEvents == 0))
            {
                Debug.LogWarning("[FaceProxyProjector] Missing required reference(s): cameraSource and/or faceProxyPrefab.");
            }
            return;
        }

        if (!cameraSource.TryGetLatestProjectionData(
                out var intrinsics,
                out var framePose,
                out long _,
                out bool intrinsicsValid,
                out bool poseValid))
        {
            _projectionDataMissCount++;
            if (enableProjectorLogs && (_projectionDataMissCount == 1 || _projectionDataMissCount % missLogEveryNEvents == 0))
            {
                Debug.LogWarning("[FaceProxyProjector] Projection data not available yet from camera source.");
            }
            return;
        }

        if (!intrinsicsValid || !poseValid)
        {
            _projectionDataMissCount++;
            if (enableProjectorLogs && (_projectionDataMissCount == 1 || _projectionDataMissCount % missLogEveryNEvents == 0))
            {
                Debug.LogWarning($"[FaceProxyProjector] Projection sample invalid. intrinsicsValid={intrinsicsValid}, poseValid={poseValid}");
            }
            return;
        }

        _projectionDataMissCount = 0;
    Transform headTransform = ResolvedHeadTransform;
    Matrix4x4 worldPose = FaceProjectionUtility.ResolveWorldPose(framePose, trackingSpaceTransform, headTransform);
        Vector3 rayOrigin = worldPose.GetPosition();

        int activeCount = Mathf.Min(maxActiveProxies, faces != null ? faces.Length : 0);
        if (enableProjectorLogs && activeCount != _lastLoggedFaceCount)
        {
            Debug.Log($"[FaceProxyProjector] Active face proxies this tick: {activeCount}");
            if (logPoseOriginDelta)
            {
                LogPoseOriginDelta(framePose);
            }
            _lastLoggedFaceCount = activeCount;
        }

        _detectionTickCount++;
        if (enableProjectorLogs && logFaceArea && activeCount > 0)
        {
            float primaryArea = GetFaceAreaRatio(faces[0]);
            _maxObservedFaceArea = Mathf.Max(_maxObservedFaceArea, primaryArea);
            _minObservedFaceArea = Mathf.Min(_minObservedFaceArea, primaryArea);

            if (_detectionTickCount == 1 || _detectionTickCount % faceAreaLogEveryNDetections == 0)
            {
                float mappedT = ComputeDepthLerpT(primaryArea);
                float previewDepth = Mathf.Lerp(Mathf.Max(minDepth, maxDepth), Mathf.Min(minDepth, maxDepth), mappedT);
                Debug.Log($"[FaceProxyProjector] FaceArea={primaryArea:F4} (minSeen={_minObservedFaceArea:F4}, maxSeen={_maxObservedFaceArea:F4}) " +
                          $"FarRef={faceAreaFar:F4}, NearRef={faceAreaNear:F4}, DepthT={mappedT:F3}, Depth={previewDepth:F3}m");
            }
        }

        EnsureCapacity(maxActiveProxies);

        for (int i = 0; i < activeCount; i++)
        {
            var face = faces[i];
            var center = new Vector2(Mathf.Clamp01(face.center.x), Mathf.Clamp01(face.center.y));
            float depth = ComputeDepth(face);

            Vector3 worldPoint = FaceProjectionUtility.ProjectViewportPointAtDepth(intrinsics, framePose, center, depth, trackingSpaceTransform, headTransform);

            ProxySlot slot = _slots[i];
            slot.rayOrigin = rayOrigin;
            slot.targetPosition = worldPoint;
            slot.lastSeenTime = Time.time;

            if (!slot.hasTarget)
            {
                slot.transform.position = worldPoint;
                slot.velocity = Vector3.zero;
            }

            slot.hasTarget = true;
            slot.transform.gameObject.SetActive(true);
            SetRenderersEnabled(slot.proxyRenderers, visualizeFaceProxies);
            if (slot.rayRenderer != null)
            {
                slot.rayRenderer.enabled = visualizeRays;
            }
        }
    }

    private void EnsureCapacity(int capacity)
    {
        while (_slots.Count < capacity)
        {
            Transform parent = proxyParent != null ? proxyParent : transform;
            GameObject instance = Instantiate(faceProxyPrefab, parent);
            instance.name = $"FaceProxy_{_slots.Count}";
            instance.SetActive(false);

            if (addGazeTargetToSpawnedProxies)
            {
                EnsureGazeTarget(instance);
            }

            _slots.Add(new ProxySlot
            {
                transform = instance.transform,
                rayRenderer = CreateRayRenderer(instance.transform),
                proxyRenderers = instance.GetComponentsInChildren<Renderer>(true),
                targetPosition = instance.transform.position,
                rayOrigin = instance.transform.position,
                velocity = Vector3.zero,
                lastSeenTime = -999f,
                hasTarget = false
            });
        }
    }

    private void EnsureGazeTarget(GameObject instance)
    {
        FaceProxyGazeTarget gazeTarget = instance.GetComponentInChildren<FaceProxyGazeTarget>();
        if (gazeTarget == null)
        {
            gazeTarget = instance.AddComponent<FaceProxyGazeTarget>();
        }

        gazeTarget.Configure(proxyDefaultMaterial, proxyHighlightMaterial);
    }

    private LineRenderer CreateRayRenderer(Transform proxyTransform)
    {
        GameObject lineObject = new GameObject("DebugRay");
        lineObject.transform.SetParent(proxyTransform, false);

        LineRenderer lineRenderer = lineObject.AddComponent<LineRenderer>();
        lineRenderer.enabled = false;
        lineRenderer.useWorldSpace = true;
        lineRenderer.positionCount = 2;
        lineRenderer.startWidth = rayWidth;
        lineRenderer.endWidth = rayWidth;
        lineRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        lineRenderer.receiveShadows = false;
        lineRenderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;

        ApplyRayRendererSettings(lineRenderer);

        return lineRenderer;
    }

    private static void SetRenderersEnabled(Renderer[] renderers, bool enabled)
    {
        if (renderers == null)
        {
            return;
        }

        for (int i = 0; i < renderers.Length; i++)
        {
            if (renderers[i] != null)
            {
                renderers[i].enabled = enabled;
            }
        }
    }

    private void ApplyRayRendererSettings(LineRenderer lineRenderer)
    {
        lineRenderer.startWidth = rayWidth;
        lineRenderer.endWidth = rayWidth;

        if (rayMaterial != null && lineRenderer.sharedMaterial != rayMaterial)
        {
            lineRenderer.sharedMaterial = rayMaterial;
        }
    }

    private float ComputeDepth(BlazeFaceDetector.DetectedFace face)
    {
        float clampedMin = Mathf.Min(minDepth, maxDepth);
        float clampedMax = Mathf.Max(minDepth, maxDepth);

        if (!useAdaptiveDepth)
        {
            return Mathf.Clamp(fixedDepthFallback, clampedMin, clampedMax);
        }

        float areaRatio = GetFaceAreaRatio(face);
        float t = ComputeDepthLerpT(areaRatio);

        // Area near faceAreaFar -> maxDepth, area near faceAreaNear -> minDepth.
        return Mathf.Lerp(clampedMax, clampedMin, t);
    }

    private float GetFaceAreaRatio(BlazeFaceDetector.DetectedFace face)
    {
        return Mathf.Clamp01(face.boundingBox.width * face.boundingBox.height);
    }

    private float ComputeDepthLerpT(float areaRatio)
    {
        float farRef = Mathf.Min(faceAreaFar, faceAreaNear);
        float nearRef = Mathf.Max(faceAreaFar, faceAreaNear);

        if (Mathf.Approximately(farRef, nearRef))
        {
            return 0f;
        }

        return Mathf.Clamp01(Mathf.InverseLerp(farRef, nearRef, areaRatio));
    }

    private void LogPoseOriginDelta(Matrix4x4 framePose)
    {
        Transform head = ResolvedHeadTransform;
        Matrix4x4 rawWorldPose = FaceProjectionUtility.ConvertTrackingPoseToWorld(framePose, trackingSpaceTransform);
        Matrix4x4 resolvedWorldPose = FaceProjectionUtility.ResolveWorldPose(framePose, trackingSpaceTransform, head);
        Vector3 rawOrigin = rawWorldPose.GetPosition();
        if (head == null)
        {
            return;
        }

        float rawDelta = Vector3.Distance(rawOrigin, head.position);
        float resolvedDelta = Vector3.Distance(resolvedWorldPose.GetPosition(), head.position);
        Debug.Log($"[FaceProxyProjector] Raw frame-pose delta: {rawDelta:F3}m, resolved ray delta: {resolvedDelta:F3}m (raw={rawOrigin}, resolved={resolvedWorldPose.GetPosition()}, head={head.position})");
    }

}
