using System.Collections;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public enum CueLateralSide
{
    Left,
    Right
}

/// <summary>
/// Displays a cue card (name, relationship, optional photo) anchored to a tracked head in world space.
///
/// Instantiation paths:
///   A) Via CueManager — call Initialize(data, headTarget, photo, audio) after AddComponent.
///   B) Standalone / test — assign the public inspector fields and set <see cref="target"/>;
///      Start() will call Initialize automatically.
/// </summary>
[RequireComponent(typeof(AudioSource))]
public class CueDisplay : MonoBehaviour
{
    [Header("Standalone / Test Mode")]
    [Tooltip("Head or face-proxy Transform to anchor the cue to (test mode only; set by CueManager at runtime).")]
    public Transform target;
    public string  personName   = "Person";
    public string  relationship = "Friend";

    [Tooltip("Optional photo shown in the lower half of the card.")]
    public Sprite    cuePhoto;
    [Tooltip("Optional audio clip played when the cue appears.")]
    public AudioClip cueAudio;

    [Tooltip("Volume applied to cue audio playback.")]
    [Range(0f, 1f)] public float cueAudioVolume = 1f;

    [Tooltip("Auto-destroy after this many seconds (0 = never).")]
    public float durationSeconds = 10f;

    [Header("Placement")]
    [Tooltip("Whether the cue appears to the left or right of the face proxy from the user's perspective.")]
    public CueLateralSide lateralSide = CueLateralSide.Right;

    [Tooltip("Horizontal distance from the proxy to the cue card.")]
    [Min(0f)] public float lateralDistance = 0.22f;

    [Tooltip("Vertical lift applied to the cue relative to the proxy or user view.")]
    public float verticalOffset = 0.05f;

    [Tooltip("If no face proxy is available, spawn this far in front of the user.")]
    [Min(0.05f)] public float fallbackDistanceFromViewer = 0.9f;

    [Tooltip("Cue will never move closer than this distance to the viewer camera.")]
    [Min(0.1f)] public float minimumDistanceFromViewer = 0.5f;

    [Tooltip("Only targets whose names start with this prefix are accepted as face proxies.")]
    public string validProxyNamePrefix = "FaceProxy_";

    [Tooltip("Known non-proxy scene object name to ignore as a target.")]
    public string blockedTargetName = "GazeAoiEngine";

    [Tooltip("Reject targets that sit at/near world origin to avoid locking onto engine roots.")]
    public bool rejectOriginTargets = true;

    [Tooltip("Radius around world origin treated as invalid target location.")]
    [Min(0f)] public float originRejectRadius = 0.05f;

    [Tooltip("Blend rate for drifting toward the target offset (higher = faster). Typical range 1–5.")]
    [Min(0f)] public float followBlendRate = 2f;

    [Tooltip("Only move when cue is farther than this distance from its desired position.")]
    [Min(0f)] public float moveThresholdDist = 0.1f;

    [Tooltip("Duration of fade-in and fade-out in seconds.")]
    [Min(0f)] public float fadeDuration = 0.35f;

    [Header("Debug")]
    [Tooltip("Log cue/proxy positions every frame for placement debugging.")]
    [SerializeField] private bool verboseDebug = true;

    [Header("Style")]
    public Vector2 panelSize = new Vector2(0.34f, 0.22f);
    public Color   panelColor = Color.white;
    public Color   textColor  = Color.black;
    public Color   lineColor  = Color.white;
    [Range(0.001f, 0.02f)] public float lineWidth = 0.004f;

    // ── Runtime state ───────────────────────────────────────────────────────────
    private Transform    _headTarget;
    private Transform    _cardRoot;   // WorldSpace Canvas root (child of this)
    private RectTransform _panelRect;
    private CanvasGroup  _canvasGroup;
    private LineRenderer _line;
    private Camera       _viewCamera;
    private AudioSource  _audioSource;
    private bool         _initialized;
    private Vector3      _lastKnownTargetPoint;
    private Coroutine    _lifecycleRoutine;
    private float        _proxyScanTimer;
    private float        _debugLogTimer;
    private int          _runtimeFontSize = 48;
    private bool         _missingRootLogged;
    private bool         _hasLastKnownTargetPoint;
    private const float  ProxyScanInterval = 0.5f;
    private const float  DebugLogInterval = 0.5f;

    // ── Unity messages ──────────────────────────────────────────────────────────

    private void Awake()
    {
        _audioSource             = GetComponent<AudioSource>();
        _audioSource.playOnAwake = false;
        _audioSource.spatialBlend = 0f; // 2-D audio — UI cue should always be clearly audible
    }

    private void Start()
    {
        // If runtime init already provided a target, do not enter standalone path.
        if (_initialized || _headTarget != null) return;

        if (target == null)
        {
            Debug.LogWarning("[CueDisplay] No target assigned and Initialize() not called - cue won't display.");
            enabled = false;
            return;
        }

        // Build from inspector fields (test / standalone mode)
        CueData testData = new CueData
        {
            font_size_px     = 48,
            duration_seconds = durationSeconds,
            cues             = new CueData.CueDetails { name = personName, relationship = relationship }
        };
        Initialize(testData, target, cuePhoto, cueAudio);
    }

    private void LateUpdate()
    {
        if (_cardRoot == null)
        {
            EnsureCardRoot();
            if (_cardRoot == null)
            {
                if (verboseDebug && !_missingRootLogged)
                {
                    _missingRootLogged = true;
                    Debug.LogWarning($"[CueDisplay] LateUpdate has no card root. children={transform.childCount}, initialized={_initialized}, enabled={enabled}");
                }
                return;
            }
        }

        // Periodically try to adopt an active proxy if we have no live target
        _proxyScanTimer -= Time.deltaTime;
        if (_proxyScanTimer <= 0f)
        {
            _proxyScanTimer = ProxyScanInterval;
            if (!HasLiveTarget())
                TryScanForProxy();
        }

        bool hasLiveTarget = HasLiveTarget();

        // When there is no live target, freeze position. "none" must not imply camera-follow motion.
        Vector3 desiredPosition = hasLiveTarget ? ComputeDesiredCardPosition() : _cardRoot.position;
        float blendRate = Mathf.Max(0.05f, followBlendRate);
        float distanceToDesired = Vector3.Distance(_cardRoot.position, desiredPosition);
        float moveThreshold = Mathf.Max(0f, moveThresholdDist);
        if (distanceToDesired > moveThreshold)
        {
            _cardRoot.position = Vector3.Lerp(_cardRoot.position, desiredPosition,
                1f - Mathf.Exp(-blendRate * Time.deltaTime));
        }

        if (hasLiveTarget)
        {
            _lastKnownTargetPoint = _headTarget.position;
            _hasLastKnownTargetPoint = true;
        }

        // Billboard: face the camera every frame. Falls back to any camera if main is null.
        Camera cam = GetViewCamera();
        if (cam != null)
        {
            ApplyBillboardRotation(cam);
        }

        UpdateConnectorLine();

        if (verboseDebug)
        {
            _debugLogTimer -= Time.deltaTime;
            if (_debugLogTimer <= 0f)
            {
                _debugLogTimer = DebugLogInterval;
                string targetInfo = HasLiveTarget()
                    ? $"{_headTarget.name} @ {_headTarget.position:F3}"
                    : "none (last known: " + _lastKnownTargetPoint.ToString("F3") + ")";
                Debug.Log($"[CueDisplay] card={_cardRoot.position:F3} desired={desiredPosition:F3} d={distanceToDesired:F3} threshold={moveThreshold:F3} target={targetInfo} cam={cam?.name} blend={blendRate:F2}");
            }
        }
    }

    public void SetVerboseDebug(bool enabled)
    {
        verboseDebug = enabled;
    }

    // ── Public API ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Builds and activates the cue card from parsed cue data.
    /// Called by <see cref="CueManager"/> immediately after AddComponent; can also be
    /// invoked manually when creating CueDisplay programmatically.
    /// </summary>
    public void Initialize(CueData data, Transform headTarget, Sprite photo = null, AudioClip audio = null)
    {
        enabled = true;
        _headTarget  = IsValidTarget(headTarget, requireActive: false) ? headTarget : null;
        target       = _headTarget;
        _viewCamera  = Camera.main;
        _lastKnownTargetPoint = _headTarget != null ? _headTarget.position : Vector3.zero;
        _hasLastKnownTargetPoint = _headTarget != null;

        personName       = data?.cues?.name         ?? "Unknown";
        relationship     = data?.cues?.relationship ?? "";
        durationSeconds  = (data != null && data.duration_seconds > 0f) ? data.duration_seconds : durationSeconds;
        cuePhoto         = photo;
        cueAudio         = audio;

        int fontSizePx = (data != null && data.font_size_px > 0) ? data.font_size_px : 48;
        _runtimeFontSize = fontSizePx;

        // Reset scan timer so first scan happens quickly after spawn
        _proxyScanTimer = 0.1f;
        _debugLogTimer = 0f;

        BuildCueCard(fontSizePx);
        BuildConnectorLine();
        EnsureCardRoot();
        SnapToInitialPosition();

        if (verboseDebug)
        {
            Debug.Log($"[CueDisplay] Post-build state: hasRoot={_cardRoot != null}, childCount={transform.childCount}");
        }

        TryPlayCueAudio();

        if (_lifecycleRoutine != null)
        {
            StopCoroutine(_lifecycleRoutine);
        }

        _lifecycleRoutine = StartCoroutine(LifecycleRoutine());

        _initialized = true;

        if (verboseDebug)
        {
            string targetName = _headTarget != null ? _headTarget.name : "none";
            Debug.Log($"[CueDisplay] Initialize complete. target={targetName}, lastKnown={_lastKnownTargetPoint:F3}, camera={GetViewCamera()?.name}");
        }
    }

    // ── Card construction ───────────────────────────────────────────────────────

    private void BuildCueCard(int fontSizePx)
    {
        // Create a fresh child GameObject as the WorldSpace Canvas root.
        // Parenting it here ensures Destroy(gameObject) cleans up the canvas too.
        GameObject root = new GameObject("CueCardRoot");
        root.transform.SetParent(transform, worldPositionStays: false);
        _cardRoot = root.transform;

        // Scale so that 1000 canvas-units == 1 world-unit (panel ends up panelSize metres)
        _cardRoot.localScale = Vector3.one * 0.001f;

        var canvas = root.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.worldCamera = _viewCamera;

        _canvasGroup = root.AddComponent<CanvasGroup>();
        _canvasGroup.alpha = 0f;

        var scaler = root.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 1000f;   // higher DPU = sharper TMP text
        root.AddComponent<GraphicRaycaster>();

        // Panel background
        GameObject panelGO = new GameObject("Panel", typeof(RectTransform), typeof(Image));
        panelGO.transform.SetParent(root.transform, false);
        _panelRect             = panelGO.GetComponent<RectTransform>();
        _panelRect.sizeDelta   = panelSize * 1000f; // canvas-units; ×0.001 scale → metres in world

        panelGO.GetComponent<Image>().color = panelColor;

        // Text — top half
        GameObject textGO = new GameObject("InfoText", typeof(RectTransform));
        textGO.transform.SetParent(panelGO.transform, false);
        var textRect       = textGO.GetComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0f, 0.5f);
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(20f,  10f);
        textRect.offsetMax = new Vector2(-20f, -10f);

        string infoText = $"This is your {relationship.ToLower()}, {personName}.";
        // string infoText = $"This is your {relationship.ToLower()}, {personName}.";
        // On some device builds, TMP settings may be missing and AddComponent<TextMeshProUGUI>() throws.
        // Fall back to UnityEngine.UI.Text so cue spawning continues.
        try
        {
            var tmp = textGO.AddComponent<TextMeshProUGUI>();
            tmp.text      = infoText;
            tmp.fontSize  = fontSizePx * 0.875f; // nominal px → TMP units
            tmp.color     = textColor;
            tmp.alignment = TextAlignmentOptions.MidlineLeft;
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"[CueDisplay] TMP unavailable at runtime; using UI.Text fallback. {ex.Message}");
            var uiText = textGO.AddComponent<Text>();
            uiText.text = infoText;
            uiText.fontSize = Mathf.RoundToInt(fontSizePx * 0.5f);
            uiText.color = textColor;
            uiText.alignment = TextAnchor.MiddleLeft;
            uiText.horizontalOverflow = HorizontalWrapMode.Wrap;
            uiText.verticalOverflow = VerticalWrapMode.Overflow;
            uiText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        }

        // Photo — bottom half
        GameObject photoGO = new GameObject("Photo", typeof(RectTransform), typeof(Image));
        photoGO.transform.SetParent(panelGO.transform, false);
        var photoRect       = photoGO.GetComponent<RectTransform>();
        photoRect.anchorMin = Vector2.zero;
        photoRect.anchorMax = new Vector2(1f, 0.5f);
        photoRect.offsetMin = Vector2.zero;
        photoRect.offsetMax = Vector2.zero;

        var photoImage           = photoGO.GetComponent<Image>();
        photoImage.sprite        = cuePhoto;
        photoImage.preserveAspect = true;
        photoImage.color         = cuePhoto == null ? new Color(0.85f, 0.85f, 0.85f, 1f) : Color.white;
    }

    private void BuildConnectorLine()
    {
        GameObject lineGO = new GameObject("CueConnectorLine");
        lineGO.transform.SetParent(transform, worldPositionStays: false);

        _line                = lineGO.AddComponent<LineRenderer>();
        _line.positionCount  = 2;
        _line.useWorldSpace  = true;
        _line.startWidth     = lineWidth;
        _line.endWidth       = lineWidth;
        _line.material       = new Material(Shader.Find("Sprites/Default"));
        _line.startColor     = lineColor;
        _line.endColor       = lineColor;
        SetVisualAlpha(0f);

        UpdateConnectorLine();
    }

    private void UpdateConnectorLine()
    {
        if (_line == null) return;

        if (!ResolvePanelRectReference())
        {
            return;
        }

        if (!HasLiveTarget())
        {
            _line.enabled = false;
            return;
        }

        _line.enabled = true;
        Vector3 targetPoint = ComputeTargetConnectionPoint();

        // Get exact side midpoint from world corners for stable connector anchoring.
        Vector3[] corners = new Vector3[4];
        _panelRect.GetWorldCorners(corners);
        Vector3 leftMid = 0.5f * (corners[0] + corners[1]);
        Vector3 rightMid = 0.5f * (corners[2] + corners[3]);
        Vector3 panelEdge = lateralSide == CueLateralSide.Right ? leftMid : rightMid;

        _line.SetPosition(0, panelEdge);
        _line.SetPosition(1, targetPoint);
    }

    private void SnapToInitialPosition()
    {
        if (_cardRoot == null) return;

        // Spawn directly at the intended lateral-offset position.
        Vector3 spawnPos;
        Transform viewer = ResolveViewerTransform();
        if (_hasLastKnownTargetPoint)
        {
            spawnPos = ComputeDesiredCardPositionFromTargetPoint(_lastKnownTargetPoint, viewer);
        }
        else
        {
            spawnPos = ComputeFallbackSpawnPosition();
        }

        _cardRoot.position = spawnPos;
        transform.position = spawnPos;

        // Apply initial billboard rotation so card never appears at default identity rotation
        Camera cam = GetViewCamera();
        if (cam != null)
        {
            ApplyBillboardRotation(cam);
        }

        if (verboseDebug)
        {
            string targetInfo = _headTarget != null ? $"{_headTarget.name} @ {_lastKnownTargetPoint:F3}" : "none";
            Debug.Log($"[CueDisplay] Spawned at {spawnPos:F3}  cam={cam?.name}  headTarget={targetInfo}");
        }

        UpdateConnectorLine();
    }

    private Vector3 ComputeDesiredCardPosition()
    {
        Transform viewer = ResolveViewerTransform();
        if (viewer == null)
        {
            if (HasLiveTarget())
            {
                return _headTarget.position + Vector3.up * verticalOffset;
            }

            // No target and no viewer: keep current/last known pose stable.
            return _cardRoot != null ? _cardRoot.position : (_lastKnownTargetPoint + Vector3.up * verticalOffset);
        }

        Vector3 horizontalForward;
        if (HasLiveTarget())
        {
            return ComputeDesiredCardPositionFromTargetPoint(_headTarget.position, viewer);
        }

        horizontalForward = GetViewerForward(viewer);
        Vector3 fallback = viewer.position + horizontalForward * fallbackDistanceFromViewer + Vector3.up * verticalOffset;
        return ConstrainRelativeToViewer(fallback, viewer);
    }

    private Vector3 ComputeDesiredCardPositionFromTargetPoint(Vector3 targetPoint, Transform viewer)
    {
        if (viewer == null)
        {
            return targetPoint + Vector3.up * verticalOffset;
        }

        Vector3 viewerToTarget = Vector3.ProjectOnPlane(targetPoint - viewer.position, Vector3.up);
        Vector3 horizontalForward = viewerToTarget.sqrMagnitude > 0.0001f
            ? viewerToTarget.normalized
            : GetViewerForward(viewer);

        Vector3 viewerRight = Vector3.Cross(Vector3.up, horizontalForward).normalized;
        Vector3 sideDirection = lateralSide == CueLateralSide.Right ? viewerRight : -viewerRight;
        Vector3 pos = targetPoint + sideDirection * lateralDistance + Vector3.up * verticalOffset;
        return ConstrainRelativeToViewer(pos, viewer);
    }

    private Vector3 ComputeFallbackSpawnPosition()
    {
        Transform viewer = ResolveViewerTransform();
        if (viewer == null)
        {
            return _cardRoot != null ? _cardRoot.position : (_hasLastKnownTargetPoint ? _lastKnownTargetPoint : Vector3.zero);
        }

        Vector3 forward = GetViewerForward(viewer);
        Vector3 pos = viewer.position + forward * fallbackDistanceFromViewer + Vector3.up * verticalOffset;
        pos = ConstrainRelativeToViewer(pos, viewer);
        return pos;
    }

    /// <summary>
    /// Returns Camera.main if available, otherwise falls back to any active camera in the scene.
    /// This is critical on ML2 where Camera.main may not be tagged correctly.
    /// </summary>
    private Camera GetViewCamera()
    {
        if (_viewCamera != null && _viewCamera.isActiveAndEnabled)
            return _viewCamera;

        _viewCamera = Camera.main;

        if (_viewCamera == null)
        {
            Camera[] all = FindObjectsByType<Camera>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            if (all.Length > 0)
            {
                _viewCamera = all[0];
                if (verboseDebug)
                    Debug.Log($"[CueDisplay] Camera.main was null; using fallback camera '{_viewCamera.name}'.");
            }
        }

        return _viewCamera;
    }

    /// <summary>
    /// Scans the scene for any active FaceProxyGazeTarget and adopts it as the head target.
    /// Called periodically from LateUpdate when no live target is available.
    /// </summary>
    private void TryScanForProxy()
    {
        FaceProxyGazeTarget[] active = FindObjectsByType<FaceProxyGazeTarget>(
            FindObjectsInactive.Exclude, FindObjectsSortMode.None);

        if (active.Length == 0) return;

        Transform best = null;
        Vector3 reference = _cardRoot != null ? _cardRoot.position : _lastKnownTargetPoint;
        float bestDist = float.MaxValue;

        for (int i = 0; i < active.Length; i++)
        {
            Transform t = active[i].transform;
            if (!IsValidTarget(t, requireActive: true))
            {
                continue;
            }

            float d = (t.position - reference).sqrMagnitude;
            if (d < bestDist)
            {
                bestDist = d;
                best = t;
            }
        }

        if (best == null)
        {
            return;
        }

        _headTarget = best;
        if (verboseDebug)
            Debug.Log($"[CueDisplay] Adopted proxy '{_headTarget.name}' @ {_headTarget.position:F3} via scan.");
    }

    private Transform ResolveViewerTransform()
    {
        return GetViewCamera()?.transform;
    }

    private static Vector3 GetViewerForward(Transform viewer)
    {
        Vector3 flatForward = Vector3.ProjectOnPlane(viewer.forward, Vector3.up);
        if (flatForward.sqrMagnitude <= 0.0001f)
        {
            flatForward = viewer.forward.sqrMagnitude > 0.0001f ? viewer.forward.normalized : Vector3.forward;
        }

        return flatForward.normalized;
    }

    private bool HasLiveTarget()
    {
        return IsValidTarget(_headTarget, requireActive: true);
    }

    public bool DebugIsInitialized()
    {
        return _initialized;
    }

    public bool DebugHasCardRoot()
    {
        if (_cardRoot == null)
        {
            TryResolveRootReferenceFast();
        }
        return _cardRoot != null;
    }

    public int DebugChildCount()
    {
        return transform.childCount;
    }

    public bool DebugIsEnabled()
    {
        return enabled && gameObject.activeInHierarchy;
    }

    public Vector3 DebugGetCardWorldPosition()
    {
        if (_cardRoot == null)
        {
            TryResolveRootReferenceFast();
        }
        return _cardRoot != null ? _cardRoot.position : transform.position;
    }

    public Vector3 DebugGetLastKnownTargetPoint()
    {
        return _lastKnownTargetPoint;
    }

    private void EnsureCardRoot()
    {
        if (_cardRoot != null) return;

        Transform found = transform.Find("CueCardRoot");
        if (found != null)
        {
            _cardRoot = found;
            ResolvePanelRectReference();
            if (verboseDebug)
            {
                Debug.Log("[CueDisplay] Recovered existing CueCardRoot child.");
            }

            // Critical: when recovering root, snap immediately so we don't spend a frame at origin.
            SnapToInitialPosition();
            return;
        }

        // Rebuild once if root is unexpectedly missing.
        BuildCueCard(_runtimeFontSize > 0 ? _runtimeFontSize : 48);
        BuildConnectorLine();
        SnapToInitialPosition();

        if (verboseDebug)
        {
            Debug.Log($"[CueDisplay] Rebuilt missing CueCardRoot. hasRoot={_cardRoot != null}, childCount={transform.childCount}");
        }
    }

    private void TryResolveRootReferenceFast()
    {
        if (_cardRoot != null) return;

        Transform found = transform.Find("CueCardRoot");
        if (found == null) return;

        _cardRoot = found;
        ResolvePanelRectReference();
    }

    private bool ResolvePanelRectReference()
    {
        if (_panelRect != null)
        {
            return true;
        }

        if (_cardRoot == null)
        {
            return false;
        }

        Transform panel = _cardRoot.Find("Panel");
        if (panel != null)
        {
            _panelRect = panel.GetComponent<RectTransform>();
        }

        if (_panelRect == null)
        {
            // Fallback only if named lookup fails.
            _panelRect = _cardRoot.GetComponentInChildren<RectTransform>();
        }

        return _panelRect != null;
    }

    private bool IsValidTarget(Transform t, bool requireActive)
    {
        if (t == null) return false;
        if (requireActive && !t.gameObject.activeInHierarchy) return false;

        if (!string.IsNullOrEmpty(blockedTargetName) && t.name == blockedTargetName) return false;

        if (rejectOriginTargets && t.position.sqrMagnitude <= originRejectRadius * originRejectRadius)
            return false;

        bool hasGazeTargetComponent = t.GetComponent<FaceProxyGazeTarget>() != null
            || t.GetComponentInParent<FaceProxyGazeTarget>() != null
            || t.GetComponentInChildren<FaceProxyGazeTarget>() != null;

        if (hasGazeTargetComponent)
            return true;

        if (!string.IsNullOrEmpty(validProxyNamePrefix) && t.name.StartsWith(validProxyNamePrefix))
            return true;

        return false;
    }

    private void ApplyBillboardRotation(Camera cam)
    {
        if (_cardRoot == null || cam == null)
            return;

        Vector3 toCamera = cam.transform.position - _cardRoot.position;
        if (toCamera.sqrMagnitude <= 0.0001f)
            return;

        // World-space Canvas front can appear mirrored with direct LookRotation(toCamera) on some setups.
        // Flipping forward here keeps text readable and stable.
        _cardRoot.rotation = Quaternion.LookRotation(-toCamera.normalized, Vector3.up);
    }

    private Vector3 ConstrainRelativeToViewer(Vector3 worldPos, Transform viewer)
    {
        if (viewer == null) return worldPos;

        Vector3 viewerForward = GetViewerForward(viewer);
        Vector3 toPos = worldPos - viewer.position;

        // Never allow the cue to move behind the user.
        float forwardDist = Vector3.Dot(toPos, viewerForward);
        if (forwardDist < 0.05f)
        {
            worldPos += viewerForward * (0.05f - forwardDist);
            toPos = worldPos - viewer.position;
        }

        float dist = toPos.magnitude;
        if (dist < minimumDistanceFromViewer)
        {
            Vector3 dir = dist > 0.0001f ? toPos / dist : viewerForward;
            worldPos = viewer.position + dir * minimumDistanceFromViewer;
        }

        return worldPos;
    }

    private Vector3 ComputeTargetConnectionPoint()
    {
        if (!HasLiveTarget())
        {
            return _hasLastKnownTargetPoint ? _lastKnownTargetPoint : (_cardRoot != null ? _cardRoot.position : transform.position);
        }

        Transform viewer = ResolveViewerTransform();
        if (viewer == null)
        {
            return _headTarget.position;
        }

        Vector3 viewerToTarget = Vector3.ProjectOnPlane(_headTarget.position - viewer.position, Vector3.up);
        Vector3 forward = viewerToTarget.sqrMagnitude > 0.0001f ? viewerToTarget.normalized : GetViewerForward(viewer);
        Vector3 viewerRight = Vector3.Cross(Vector3.up, forward).normalized;
        Vector3 sideDir = lateralSide == CueLateralSide.Right ? viewerRight : -viewerRight;

        float sideExtent = 0.04f;
        Renderer r = _headTarget.GetComponentInChildren<Renderer>();
        if (r != null)
        {
            Vector3 e = r.bounds.extents;
            sideExtent = Mathf.Max(sideExtent, Mathf.Abs(sideDir.x) * e.x * 0.5f + Mathf.Abs(sideDir.y) * e.y * 0.5f + Mathf.Abs(sideDir.z) * e.z * 0.5f);
        }

        Collider c = _headTarget.GetComponentInChildren<Collider>();
        if (c != null)
        {
            Vector3 e = c.bounds.extents;
            sideExtent = Mathf.Max(sideExtent, Mathf.Abs(sideDir.x) * e.x * 0.5f + Mathf.Abs(sideDir.y) * e.y * 0.5f + Mathf.Abs(sideDir.z) * e.z * 0.5f);
        }

        return _headTarget.position + sideDir * sideExtent;
    }

    private IEnumerator LifecycleRoutine()
    {
        if (fadeDuration > 0f)
        {
            yield return FadeRoutine(0f, 1f, fadeDuration);
        }
        else
        {
            SetVisualAlpha(1f);
        }

        if (durationSeconds <= 0f)
        {
            yield break;
        }

        float visibleDuration = Mathf.Max(0f, durationSeconds - (fadeDuration * 2f));
        if (visibleDuration > 0f)
        {
            yield return new WaitForSeconds(visibleDuration);
        }

        if (fadeDuration > 0f)
        {
            yield return FadeRoutine(1f, 0f, fadeDuration);
        }

        Destroy(gameObject);
    }

    private IEnumerator FadeRoutine(float fromAlpha, float toAlpha, float duration)
    {
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = duration <= 0f ? 1f : Mathf.Clamp01(elapsed / duration);
            SetVisualAlpha(Mathf.Lerp(fromAlpha, toAlpha, t));
            yield return null;
        }

        SetVisualAlpha(toAlpha);
    }

    private void SetVisualAlpha(float alpha)
    {
        if (_canvasGroup != null)
        {
            _canvasGroup.alpha = alpha;
        }

        if (_line != null)
        {
            Color fadedLineColor = lineColor;
            fadedLineColor.a *= alpha;
            _line.startColor = fadedLineColor;
            _line.endColor = fadedLineColor;
        }
    }

    private void TryPlayCueAudio()
    {
        if (cueAudio == null)
        {
            if (verboseDebug)
            {
                Debug.Log("[CueDisplay] No cue audio assigned; skipping audio playback.");
            }
            return;
        }

        if (_audioSource == null)
        {
            _audioSource = GetComponent<AudioSource>();
        }

        if (_audioSource == null)
        {
            Debug.LogWarning("[CueDisplay] Missing AudioSource; cannot play cue audio.");
            return;
        }

        _audioSource.enabled = true;
        _audioSource.playOnAwake = false;
        _audioSource.spatialBlend = 0f;
        _audioSource.loop = false;
        _audioSource.volume = cueAudioVolume;

        _audioSource.Stop();
        _audioSource.clip = cueAudio;
        _audioSource.Play();

        // Fallback path for platforms where clip playback may not start on first Play() call.
        if (!_audioSource.isPlaying)
        {
            _audioSource.PlayOneShot(cueAudio, cueAudioVolume);
        }

        if (verboseDebug)
        {
            Debug.Log($"[CueDisplay] Audio playback requested: clip={cueAudio.name}, length={cueAudio.length:F2}s, isPlaying={_audioSource.isPlaying}");
        }
    }

    private void OnDisable()
    {
        if (_lifecycleRoutine != null)
        {
            StopCoroutine(_lifecycleRoutine);
            _lifecycleRoutine = null;
        }
    }
}