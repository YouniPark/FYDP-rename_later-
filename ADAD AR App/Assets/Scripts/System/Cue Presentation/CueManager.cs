using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Receives cue data from the backend (or a local JSON config) and spawns a
/// <see cref="CueDisplay"/> anchored to the configured head target.
///
/// Flow (network mode):
///   1. Polls GET backendBaseUrl + cueEndpoint every pollIntervalSeconds.
///   2. Backend responds with a JSON body matching <see cref="CueData"/>.
///   3. If the payload is for a new person (people_id changed), the old cue is
///      dismissed and a new CueDisplay is spawned.
///   4. If media override assets are not assigned, optional image_url/audio_url
///      are downloaded before the cue appears.
///
/// Flow (default config mode):
///   Toggle <see cref="useDefaultConfig"/> in the Inspector.  The JSON at
///   Resources/<see cref="defaultCueResourcePath"/>.json is loaded at Start.
/// </summary>
public class CueManager : MonoBehaviour
{
    [Header("Lifecycle")]
    [Tooltip("When disabled, CueManager will not start default/backend mode on Start(). Use TriggerDefaultCueIfNone() externally instead.")]
    [SerializeField] private bool autoStartOnPlay = true;

    [Tooltip("Emit detailed cue spawn diagnostics (target, positions, chosen source).")]
    [SerializeField] private bool verboseCueLogs = false;

    [Tooltip("Enable verbose runtime position logs inside each spawned CueDisplay.")]
    [SerializeField] private bool cueDisplayVerboseLogs = false;

    // ── Mode ────────────────────────────────────────────────────────────────────
    [Header("Mode")]
    [Tooltip("When enabled, loads and displays the local cue JSON from Resources/ instead of polling the backend.")]
    [SerializeField] private bool useDefaultConfig = false;

    [Tooltip("Path inside Resources/ (no extension) to the local cue JSON.")]
    [SerializeField] private string defaultCueResourcePath = "Cue Defaults/default_cue_daniel";

    [Tooltip("Format string for per-person cue JSONs inside Resources/. Use {0} as the people_id placeholder. E.g. 'Cue Defaults/{0}/default_cue_{0}' resolves to 'Cue Defaults/1/default_cue_1' for id=1.")]
    [SerializeField] private string perPersonCuePathFormat = "Cue Defaults/{0}/default_cue_{0}";

    // ── Network ─────────────────────────────────────────────────────────────────
    [Header("Network")]
    [Tooltip("Base URL of the backend server")]
    [SerializeField] private string backendBaseUrl = "http://10.0.0.1:5000";

    [Tooltip("Endpoint appended to backendBaseUrl to retrieve the latest cue.")]
    [SerializeField] private string cueEndpoint = "/cue";

    [Tooltip("Seconds between backend polls. Has no effect in default-config mode.")]
    [SerializeField] private float pollIntervalSeconds = 3f;

    // ── Anchor ──────────────────────────────────────────────────────────────────
    [Header("Cue Anchor")]
    [Tooltip("Optional explicit anchor. If left null, CueManager auto-selects any spawned FaceProxy at runtime.")]
    [SerializeField] private Transform cueTarget;

    [Tooltip("When Cue Target is empty, automatically select an available spawned face proxy.")]
    [SerializeField] private bool autoFindFaceProxyTarget = true;

    [Tooltip("Fallback name prefix used to detect spawned proxies when FaceProxyGazeTarget is not present.")]
    [SerializeField] private string faceProxyNamePrefix = "FaceProxy_";

    [Tooltip("Scene object name that should never be treated as a face proxy target.")]
    [SerializeField] private string blockedTargetName = "GazeAoiEngine";

    [Tooltip("Reject targets that sit at/near world origin to avoid engine-root false positives.")]
    [SerializeField] private bool rejectOriginTargets = true;

    [Tooltip("Radius around world origin treated as invalid target location.")]
    [SerializeField] private float originRejectRadius = 0.05f;

    // ── Media sources ──────────────────────────────────────────────────────────
    [Header("Media Sources")]
    [Tooltip("If enabled and Cue Photo Override is assigned, that Sprite is used instead of cues.image_url.")]
    [SerializeField] private bool overrideImageFromInspector = true;

    [Tooltip("Drag a local Sprite asset here (e.g. from Assets/Resources/Cue Defaults) to bypass image URL download.")]
    [SerializeField] private Sprite cuePhotoOverride;

    [Tooltip("If enabled and Cue Audio Override is assigned, that clip is used instead of cues.audio_url.")]
    [SerializeField] private bool overrideAudioFromInspector = true;

    [Tooltip("Drag a local AudioClip asset here (e.g. from Assets/Resources/Cue Defaults) to bypass audio URL download.")]
    [SerializeField] private AudioClip cueAudioOverride;

    // ── Appearance overrides ────────────────────────────────────────────────────
    [Header("Placement (overrides CueDisplay defaults)")]
    [SerializeField] private CueLateralSide lateralSide = CueLateralSide.Right;
    [SerializeField] private float lateralDistance = 0.22f;
    [SerializeField] private float verticalOffset = 0.05f;
    [SerializeField] private float fallbackDistanceFromViewer = 0.9f;
    [SerializeField] private float minimumDistanceFromViewer = 0.5f;
    [SerializeField] private float followBlendRate = 2f;
    [SerializeField] private float moveThresholdDist = 0.1f;
    [SerializeField] private float fadeDuration = 0.35f;

    [Header("Appearance (overrides CueDisplay defaults)")]
    [SerializeField] private Vector2 panelSize   = new Vector2(0.34f, 0.22f);
    [SerializeField] private Color   panelColor  = Color.black;
    [SerializeField] private Color   textColor   = Color.white;
    [SerializeField] private Color   lineColor   = Color.white;
    [Range(0.001f, 0.02f)]
    [SerializeField] private float   lineWidth   = 0.004f;

    // ── State ───────────────────────────────────────────────────────────────────
    private CueDisplay _activeCue;
    private int        _lastDisplayedId = -1;
    private bool       _spawnRequestInFlight;

    public bool HasActiveCue => _activeCue != null;
    public int CurrentPeopleId => _lastDisplayedId;

    // ── Unity messages ──────────────────────────────────────────────────────────

    private void Start()
    {
        if (!autoStartOnPlay)
            return; // do nothing until triggered

        if (useDefaultConfig)
            LoadAndShowDefaultCue();
        else
            StartCoroutine(PollBackendRoutine());
    }

    // ── Default config ──────────────────────────────────────────────────────────

    private void LoadAndShowDefaultCue()
    {
        if (!TryLoadDefaultCueData(out CueData data))
        {
            return; // do nothing if default cue data is not available
        }

        StartCoroutine(ShowCueRoutine(data));
    }

    /// <summary>
    /// External entry point for local testing or connection managers.
    /// Spawns a cue from local JSON config only when no cue is currently active.
    /// <paramref name="hintTarget"/> is the preferred anchor (e.g. the exact proxy the user fixated);
    /// if null, ResolveCueTarget() is used as usual.
    /// </summary>
    public bool TriggerCueIfNone(Transform hintTarget = null)
    {
        if (HasActiveCue || _spawnRequestInFlight)
        {
            return false; // do nothing if a cue is already active or a spawn request is in flight
        }

        if (!TryLoadDefaultCueData(out CueData data))
        {
            return false; // do nothing if default cue data is not available
        }

        StartCoroutine(ShowCueIfNoneRoutine(data, hintTarget));
        return true;
    }

    // Backwards-compat wrapper for existing callers.
    public bool TriggerDefaultCueIfNone(Transform hintTarget = null)
    {
        return TriggerCueIfNone(hintTarget);
    }

    /// <summary>
    /// Spawns the cue for a specific person ID, loading from the per-person JSON path format.
    /// Dismisses any active cue first. Safe to call even when a cue is already showing.
    /// </summary>
    public bool TriggerCueForPerson(int peopleId, Transform hintTarget = null)
    {
        if (_spawnRequestInFlight)
        {
            return false;
        }

        string path = string.Format(perPersonCuePathFormat, peopleId);
        TextAsset json = Resources.Load<TextAsset>(path);
        if (json == null)
        {
            Debug.LogWarning($"[CueManager] Per-person cue JSON not found at Resources/{path} (peopleId={peopleId}).");
            return false;
        }

        CueData data = JsonUtility.FromJson<CueData>(json.text);
        if (data == null || data.cues == null)
        {
            Debug.LogError($"[CueManager] Failed to parse cue JSON at Resources/{path}.");
            return false;
        }

        // Dismiss existing cue and spawn the new one immediately.
        if (_activeCue != null)
        {
            Destroy(_activeCue.gameObject);
            _activeCue = null;
        }

        StartCoroutine(ShowCueIfNoneRoutine(data, hintTarget));
        return true;
    }

    // ── Network polling ─────────────────────────────────────────────────────────

    private IEnumerator PollBackendRoutine()
    {
        while (true)
        {
            yield return StartCoroutine(FetchAndShowCue());
            yield return new WaitForSeconds(pollIntervalSeconds);
        }
    }

    private IEnumerator ShowCueIfNoneRoutine(CueData data, Transform hintTarget = null)
    {
        if (HasActiveCue)
        {
            yield break;
        }

        _spawnRequestInFlight = true;
        yield return StartCoroutine(ShowCueRoutine(data, hintTarget));
        _spawnRequestInFlight = false;
    }

    private IEnumerator FetchAndShowCue()
    {
        string url = backendBaseUrl.TrimEnd('/') + cueEndpoint;
        using (UnityWebRequest req = UnityWebRequest.Get(url))
        {
            req.timeout = 5;
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning($"[CueManager] Backend request failed ({url}): {req.error}");
                yield break;
            }

            CueData data = JsonUtility.FromJson<CueData>(req.downloadHandler.text);
            if (data == null || data.cues == null)
            {
                Debug.LogWarning("[CueManager] Could not parse cue payload from backend.");
                yield break;
            }

            // Don't re-show the same person if they're already on screen
            if (_activeCue != null && data.people_id == _lastDisplayedId)
                yield break;

            yield return StartCoroutine(ShowCueRoutine(data));
        }
    }

    // ── Cue spawning ────────────────────────────────────────────────────────────

    private IEnumerator ShowCueRoutine(CueData data, Transform hintTarget = null)
    {
        // Dismiss any existing cue first
        if (_activeCue != null)
        {
            Destroy(_activeCue.gameObject);
            _activeCue = null;
        }

        // Resolve media — priority: inspector override > JSON image_path/audio_path (Resources.Load) > URL download.
        Sprite photo = null;
        if (overrideImageFromInspector && cuePhotoOverride != null)
        {
            photo = cuePhotoOverride;
            if (verboseCueLogs) Debug.Log($"[CueManager] Image: inspector override '{cuePhotoOverride.name}'.");
        }
        else if (!string.IsNullOrEmpty(data.cues?.image_path))
        {
            Texture2D tex = Resources.Load<Texture2D>(data.cues.image_path);
            if (tex != null)
            {
                photo = Sprite.Create(tex, new Rect(0f, 0f, tex.width, tex.height), new Vector2(0.5f, 0.5f), 100f);
                if (verboseCueLogs) Debug.Log($"[CueManager] Image: loaded from Resources '{data.cues.image_path}'.");
            }
            else
            {
                Debug.LogWarning($"[CueManager] Image: Resources.Load failed for '{data.cues.image_path}'.");
            }
        }
        else if (!string.IsNullOrEmpty(data.cues?.image_url))
        {
            yield return StartCoroutine(DownloadImage(data.cues.image_url, sprite => photo = sprite));
            if (verboseCueLogs) Debug.Log($"[CueManager] Image: downloaded from '{data.cues.image_url}'.");
        }
        else if (verboseCueLogs)
        {
            Debug.Log("[CueManager] Image: no source specified.");
        }

        AudioClip audio = null;
        if (overrideAudioFromInspector && cueAudioOverride != null)
        {
            audio = cueAudioOverride;
            if (verboseCueLogs) Debug.Log($"[CueManager] Audio: inspector override '{audio.name}'.");
        }
        else if (!string.IsNullOrEmpty(data.cues?.audio_path))
        {
            audio = Resources.Load<AudioClip>(data.cues.audio_path);
            if (audio != null)
            {
                if (verboseCueLogs) Debug.Log($"[CueManager] Audio: loaded from Resources '{data.cues.audio_path}'.");
            }
            else
            {
                Debug.LogWarning($"[CueManager] Audio: Resources.Load failed for '{data.cues.audio_path}'.");
            }
        }
        else if (!string.IsNullOrEmpty(data.cues?.audio_url))
        {
            yield return StartCoroutine(DownloadAudio(data.cues.audio_url, clip => audio = clip));
            if (verboseCueLogs) Debug.Log($"[CueManager] Audio: downloaded from '{data.cues.audio_url}'.");
        }
        else if (verboseCueLogs)
        {
            Debug.Log("[CueManager] Audio: no source specified.");
        }

        // Spawn the cue display object
        GameObject go  = new GameObject("CueDisplay");
        CueDisplay cue = go.AddComponent<CueDisplay>();

        // Prefer the hint target (e.g. exact proxy from fixation event), then fall back to scene search.
        Transform resolvedTarget = (IsValidFaceProxyTransform(hintTarget, requireActive: true))
            ? hintTarget
            : ResolveCueTarget();

        if (resolvedTarget == null)
        {
            Debug.LogWarning("[CueManager] No face proxy target found yet; cue will spawn without following a proxy.");
        }

        // Apply manager-level appearance settings
        cue.lateralSide = lateralSide;
        cue.lateralDistance = lateralDistance;
        cue.verticalOffset = verticalOffset;
        cue.fallbackDistanceFromViewer = fallbackDistanceFromViewer;
        cue.minimumDistanceFromViewer = minimumDistanceFromViewer;
        cue.validProxyNamePrefix = faceProxyNamePrefix;
        cue.blockedTargetName = blockedTargetName;
        cue.rejectOriginTargets = rejectOriginTargets;
        cue.originRejectRadius = originRejectRadius;
        cue.followBlendRate = followBlendRate;
        cue.moveThresholdDist = moveThresholdDist;
        cue.fadeDuration = fadeDuration;
        cue.panelSize   = panelSize;
        cue.panelColor  = panelColor;
        cue.textColor   = textColor;
        cue.lineColor   = lineColor;
        cue.lineWidth   = lineWidth;
        cue.SetVerboseDebug(cueDisplayVerboseLogs);

        if (verboseCueLogs)
        {
            string targetText = resolvedTarget != null
                ? $"{resolvedTarget.name} @ {resolvedTarget.position:F3}"
                : "none";
            Debug.Log($"[CueManager] Spawn cue request: person={data.people_id}, hint={hintTarget?.name}, resolvedTarget={targetText}, followBlend={followBlendRate:F2}, moveThreshold={moveThresholdDist:F2}");
        }

        cue.Initialize(data, resolvedTarget, photo: photo, audio: audio);

        if (verboseCueLogs)
        {
            Debug.Log($"[CueManager] Cue initialized: cueObj={cue.name}, goPos={(cue.transform != null ? cue.transform.position.ToString("F3") : "n/a")}, cardPos={cue.DebugGetCardWorldPosition():F3}, initialized={cue.DebugIsInitialized()}, enabled={cue.DebugIsEnabled()}, hasRoot={cue.DebugHasCardRoot()}, children={cue.DebugChildCount()}, target={resolvedTarget?.name}");
            StartCoroutine(LogCueStateNextFrames(cue, resolvedTarget));
        }

        _activeCue       = cue;
        _lastDisplayedId = data.people_id;
    }

    // ── Media download ──────────────────────────────────────────────────────────

    private IEnumerator DownloadImage(string url, System.Action<Sprite> callback)
    {
        using (UnityWebRequest req = UnityWebRequestTexture.GetTexture(url))
        {
            req.timeout = 10;
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning($"[CueManager] Image download failed ({url}): {req.error}");
                callback(null);
                yield break;
            }

            Texture2D tex = DownloadHandlerTexture.GetContent(req);
            if (tex == null)
            {
                callback(null);
                yield break;
            }

            Sprite sprite = Sprite.Create(
                tex,
                new Rect(0f, 0f, tex.width, tex.height),
                new Vector2(0.5f, 0.5f),
                100f);
            callback(sprite);
        }
    }

    private IEnumerator DownloadAudio(string url, System.Action<AudioClip> callback)
    {
        AudioType audioType = InferAudioType(url);
        if (audioType == AudioType.UNKNOWN)
        {
            Debug.LogWarning($"[CueManager] Unrecognised audio format for '{url}'; skipping.");
            callback(null);
            yield break;
        }

        using (UnityWebRequest req = UnityWebRequestMultimedia.GetAudioClip(url, audioType))
        {
            req.timeout = 10;
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogWarning($"[CueManager] Audio download failed ({url}): {req.error}");
                callback(null);
                yield break;
            }

            callback(DownloadHandlerAudioClip.GetContent(req));
        }
    }

    private static AudioType InferAudioType(string url)
    {
        string lower = url.ToLowerInvariant();
        if (lower.EndsWith(".mp3")) return AudioType.MPEG;
        if (lower.EndsWith(".ogg")) return AudioType.OGGVORBIS;
        if (lower.EndsWith(".wav")) return AudioType.WAV;
        return AudioType.UNKNOWN;
    }

    private bool TryLoadDefaultCueData(out CueData data)
    {
        data = null;
        TextAsset json = Resources.Load<TextAsset>(defaultCueResourcePath);
        if (json == null)
        {
            Debug.LogError($"[CueManager] Default cue JSON not found at Resources/{defaultCueResourcePath}");
            return false;
        }

        data = JsonUtility.FromJson<CueData>(json.text);
        if (data == null || data.cues == null)
        {
            Debug.LogError("[CueManager] Failed to parse default cue JSON.");
            return false;
        }

        return true;
    }

    private Transform ResolveCueTarget()
    {
        if (IsValidFaceProxyTransform(cueTarget, requireActive: false))
        {
            return cueTarget;
        }

        if (!autoFindFaceProxyTarget)
        {
            return null;
        }

        // Preferred: proxies with FaceProxyGazeTarget (added by FaceProxyProjector by default).
        FaceProxyGazeTarget[] gazeTargets = FindObjectsByType<FaceProxyGazeTarget>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None);

        Transform firstAny = null;
        for (int i = 0; i < gazeTargets.Length; i++)
        {
            Transform t = gazeTargets[i] != null ? gazeTargets[i].transform : null;
            if (!IsValidFaceProxyTransform(t, requireActive: false)) continue;

            if (firstAny == null)
            {
                firstAny = t;
            }

            if (t.gameObject.activeInHierarchy)
            {
                return t;
            }
        }

        if (firstAny != null)
        {
            return firstAny;
        }

        // Fallback: by generated name prefix from FaceProxyProjector (FaceProxy_0, ...)
        Transform[] allTransforms = FindObjectsByType<Transform>(
            FindObjectsInactive.Include,
            FindObjectsSortMode.None);

        Transform firstNamedAny = null;
        for (int i = 0; i < allTransforms.Length; i++)
        {
            Transform t = allTransforms[i];
            if (!IsValidFaceProxyTransform(t, requireActive: false)) continue;

            if (firstNamedAny == null)
            {
                firstNamedAny = t;
            }

            if (t.gameObject.activeInHierarchy)
            {
                return t;
            }
        }

        return firstNamedAny;
    }

    private bool IsValidFaceProxyTransform(Transform t, bool requireActive)
    {
        if (t == null) return false;
        if (requireActive && !t.gameObject.activeInHierarchy) return false;
        if (!string.IsNullOrEmpty(blockedTargetName) && t.name == blockedTargetName) return false;

        if (rejectOriginTargets && t.position.sqrMagnitude <= originRejectRadius * originRejectRadius)
            return false;

        bool hasGazeTarget = t.GetComponent<FaceProxyGazeTarget>() != null
            || t.GetComponentInParent<FaceProxyGazeTarget>() != null
            || t.GetComponentInChildren<FaceProxyGazeTarget>() != null;

        if (hasGazeTarget)
            return true;

        if (!string.IsNullOrEmpty(faceProxyNamePrefix) && t.name.StartsWith(faceProxyNamePrefix))
            return true;

        return false;
    }

    private IEnumerator LogCueStateNextFrames(CueDisplay cue, Transform target)
    {
        if (!verboseCueLogs || cue == null) yield break;

        yield return null; // next frame
        if (cue != null)
        {
            Debug.Log($"[CueManager] Cue +1f: cardPos={cue.DebugGetCardWorldPosition():F3}, lastKnown={cue.DebugGetLastKnownTargetPoint():F3}, initialized={cue.DebugIsInitialized()}, enabled={cue.DebugIsEnabled()}, hasRoot={cue.DebugHasCardRoot()}, children={cue.DebugChildCount()}, targetAlive={(target != null && target.gameObject.activeInHierarchy)}");
        }

        yield return new WaitForSeconds(0.5f);
        if (cue != null)
        {
            Debug.Log($"[CueManager] Cue +0.5s: cardPos={cue.DebugGetCardWorldPosition():F3}, lastKnown={cue.DebugGetLastKnownTargetPoint():F3}, initialized={cue.DebugIsInitialized()}, enabled={cue.DebugIsEnabled()}, hasRoot={cue.DebugHasCardRoot()}, children={cue.DebugChildCount()}, targetAlive={(target != null && target.gameObject.activeInHierarchy)}");
        }
    }
}
