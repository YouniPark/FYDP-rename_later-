using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Basic connection bridge for local integration testing.
///
/// Current behavior:
/// - Listens to FaceProxyGazeInteractor fixation events.
/// - Skips backend/LSL and asks CueManager to spawn a default cue.
/// - Will only request a spawn when no cue is currently active.
///
/// This creates a clean insertion point for future LSL/backend output without
/// modifying the gaze interaction logic again.
/// </summary>
public class CueConnectionManager : MonoBehaviour
{
    public enum CueRoutingMode
    {
        LocalDefault = 0,
        LocalCycle = 1,
        BackendFaceLookup = 2,
    }

    [System.Serializable]
    private class BackendLatestFaceResponseDto
    {
        public string name;
        public int people_id;
        public float confidence;
        public string decided_at;
        public string source;
        public float window_seconds;
        public int sample_count;
        public bool is_unknown;
    }

    [Header("References")]
    [SerializeField] private FaceProxyGazeInteractor gazeInteractor;
    [SerializeField] private CueManager cueManager;

    [Header("Routing")]
    [Tooltip("When enabled, fixation events directly trigger a default cue spawn request.")]
    [SerializeField] private bool spawnCueDirectlyOnFixation = true;

    [Tooltip("Select how fixation events map to cue spawning.")]
    [SerializeField] private CueRoutingMode routingMode = CueRoutingMode.LocalDefault;

    [Header("Timing")]
    [SerializeField, Min(0f), Tooltip("Delay between fixation event and cue routing. Set to 0 for immediate response.")]
    private float fixationToCueDelaySeconds = 0.5f;

    [Tooltip("Number of people to cycle through when Routing Mode is LocalCycle.")]
    [SerializeField] private int totalPeople = 4;

    [Header("Backend Lookup")]
    [Tooltip("Lookup configuration for backend-driven cue routing.")]
    [SerializeField] private BackendFaceLookupConfig backendLookupConfig = new BackendFaceLookupConfig();

    [SerializeField, Tooltip("Preferred shared backend address source. Assign the BackendConnectionConfig component here. If left empty, the first one found in scene is used, then legacy URL fallbacks are used.")]
    private MonoBehaviour sharedBackendConfigComponent;

    [Tooltip("Log routing decisions for debugging while wiring connections.")]
    [SerializeField] private bool verboseLogs = true;

    private int _cycleIndex = 0;
    private bool _backendLookupInFlight;

    private void Awake()
    {
        if (sharedBackendConfigComponent == null)
        {
            sharedBackendConfigComponent = FindSharedBackendConfigComponent();
        }

        if (gazeInteractor == null)
        {
            gazeInteractor = FindFirstObjectByType<FaceProxyGazeInteractor>();
        }

        if (cueManager == null)
        {
            cueManager = FindFirstObjectByType<CueManager>();
        }
    }

    private void OnEnable()
    {
        if (gazeInteractor != null)
        {
            gazeInteractor.OnFixationEvent += HandleFixationEvent;
        }
        else if (verboseLogs)
        {
            Debug.LogWarning("[CueConnectionManager] FaceProxyGazeInteractor reference missing; no fixation events will be received.");
        }
    }

    private void OnDisable()
    {
        if (gazeInteractor != null)
        {
            gazeInteractor.OnFixationEvent -= HandleFixationEvent;
        }

        StopAllCoroutines();
        _backendLookupInFlight = false;
    }

    private void HandleFixationEvent(FaceProxyGazeTarget target)
    {
        if (!spawnCueDirectlyOnFixation)
        {
            return;
        }

        if (cueManager == null)
        {
            if (verboseLogs)
            {
                Debug.LogWarning("[CueConnectionManager] CueManager reference missing; cannot spawn cue.");
            }
            return;
        }

        // Pass the exact fixated proxy so the cue anchors to the right face immediately.
        Transform hintTarget = target != null ? target.transform : null;

        StartCoroutine(HandleFixationEventAfterDelay(target, hintTarget));
    }

    private IEnumerator HandleFixationEventAfterDelay(FaceProxyGazeTarget target, Transform hintTarget)
    {
        float delaySeconds = Mathf.Max(0f, fixationToCueDelaySeconds);
        if (delaySeconds > 0f)
        {
            yield return new WaitForSeconds(delaySeconds);
        }

        switch (routingMode)
        {
            case CueRoutingMode.LocalCycle:
                HandleLocalCycleRoute(target, hintTarget);
                yield break;

            case CueRoutingMode.BackendFaceLookup:
                if (_backendLookupInFlight)
                {
                    if (verboseLogs)
                    {
                        string targetName = target != null ? target.name : "null";
                        Debug.Log($"[CueConnectionManager] Fixation on {targetName} ignored: backend lookup already in flight.");
                    }
                    yield break;
                }

                StartCoroutine(HandleBackendRouteCoroutine(target, hintTarget));
                yield break;

            default:
                HandleLocalDefaultRoute(target, hintTarget);
                yield break;
        }
    }

    private void HandleLocalCycleRoute(FaceProxyGazeTarget target, Transform hintTarget)
    {
        // Advance to the next person and present a new cue only if its people_id differs
        // from the cue currently on screen. This mirrors the server-side face-id change logic.
        int count = Mathf.Max(1, totalPeople);
        int nextPeopleId = (_cycleIndex % count) + 1; // cycles 1 -> 2 -> ... -> totalPeople -> 1

        if (cueManager.CurrentPeopleId == nextPeopleId)
        {
            nextPeopleId = (nextPeopleId % count) + 1;
        }

        bool started = cueManager.TriggerCueForPerson(nextPeopleId, hintTarget);
        if (started)
        {
            _cycleIndex = nextPeopleId;
        }

        if (verboseLogs)
        {
            string targetName = target != null ? target.name : "null";
            Debug.Log(started
                ? $"[CueConnectionManager] Fixation on {targetName} -> cycling to people_id {nextPeopleId} / {count} (current={cueManager.CurrentPeopleId})."
                : $"[CueConnectionManager] Fixation on {targetName} -> TriggerCueForPerson({nextPeopleId}) not started.");
        }
    }

    private void HandleLocalDefaultRoute(FaceProxyGazeTarget target, Transform hintTarget)
    {

        if (cueManager.HasActiveCue)
        {
            if (verboseLogs)
            {
                string targetName = target != null ? target.name : "null";
                Debug.Log($"[CueConnectionManager] Fixation on {targetName}, but cue already active. Skipping spawn.");
            }
            return;
        }

        bool defaultStarted = cueManager.TriggerCueIfNone(hintTarget);
        if (verboseLogs)
        {
            string targetName = target != null ? target.name : "null";
            Debug.Log(defaultStarted
                ? $"[CueConnectionManager] Fixation on {targetName} -> requested cue spawn."
                : $"[CueConnectionManager] Fixation on {targetName} -> spawn request not started.");
        }
    }

    private IEnumerator HandleBackendRouteCoroutine(FaceProxyGazeTarget target, Transform hintTarget)
    {
        _backendLookupInFlight = true;
        try
        {
            string url = BuildLatestFaceUrl();
            using (UnityWebRequest req = UnityWebRequest.Get(url))
            {
                req.timeout = Mathf.Max(1, Mathf.RoundToInt(backendLookupConfig.requestTimeoutSeconds));
                yield return req.SendWebRequest();

                if (req.result != UnityWebRequest.Result.Success)
                {
                    if (verboseLogs)
                    {
                        Debug.LogWarning($"[CueConnectionManager] Backend lookup failed ({url}): {req.error}");
                    }

                    yield break;
                }

                string json = req.downloadHandler.text;
                bool peopleIdWasNull = json.Contains("\"people_id\":null");

                BackendLatestFaceResponseDto latest = JsonUtility.FromJson<BackendLatestFaceResponseDto>(json);
                if (latest == null)
                {
                    if (verboseLogs)
                    {
                        Debug.LogWarning("[CueConnectionManager] Backend lookup parse failed: null response DTO.");
                    }

                    yield break;
                }

                string targetName = target != null ? target.name : "null";

                // No-face means backend has no decision yet; do not spawn cues.
                if (peopleIdWasNull)
                {
                    if (verboseLogs)
                    {
                        Debug.Log($"[CueConnectionManager] Fixation on {targetName} -> backend returned no-face (people_id=null). Skipping cue.");
                    }

                    yield break;
                }

                // Unknown, currently doesn't produce a cue spawn.
                if (latest.is_unknown || latest.people_id == 0)
                {
                    if (verboseLogs)
                    {
                        Debug.Log($"[CueConnectionManager] Fixation on {targetName} -> backend returned Unknown. Skipping cue.");
                    }

                    yield break;
                }

                if (latest.people_id <= 0 || latest.people_id >= 4)
                {
                    if (verboseLogs)
                    {
                        Debug.LogWarning($"[CueConnectionManager] Fixation on {targetName} -> invalid people_id={latest.people_id}. Skipping cue.");
                    }

                    yield break;
                }

                if (cueManager.CurrentPeopleId == latest.people_id)
                {
                    if (verboseLogs)
                    {
                        Debug.Log($"[CueConnectionManager] Fixation on {targetName} -> people_id {latest.people_id} already active. No respawn.");
                    }

                    yield break;
                }

                bool started = cueManager.TriggerCueForPerson(latest.people_id, hintTarget);
                if (verboseLogs)
                {
                    Debug.Log(started
                        ? $"[CueConnectionManager] Fixation on {targetName} -> backend people_id {latest.people_id}, confidence={latest.confidence:F2}, cue requested."
                        : $"[CueConnectionManager] Fixation on {targetName} -> TriggerCueForPerson({latest.people_id}) not started.");
                }
            }
        }
        finally
        {
            _backendLookupInFlight = false;
        }
    }

    private string BuildLatestFaceUrl()
    {
        string sharedUrl = InvokeSharedBackendStringMethod("BuildFaceLookupUrl");
        if (!string.IsNullOrWhiteSpace(sharedUrl))
        {
            return sharedUrl.Trim();
        }

        string configuredUrl = backendLookupConfig != null ? backendLookupConfig.faceLookupUrl : null;
        if (!string.IsNullOrWhiteSpace(configuredUrl))
        {
            return configuredUrl.Trim();
        }

        return "http://127.0.0.1:8001/face/latest";
    }

    public string BuildVideoWebSocketUrl()
    {
        string sharedUrl = InvokeSharedBackendStringMethod("BuildVideoWebSocketUri");
        if (!string.IsNullOrWhiteSpace(sharedUrl))
        {
            return sharedUrl.Trim();
        }

        string configuredUrl = backendLookupConfig != null ? backendLookupConfig.videoWsUrl : null;
        if (!string.IsNullOrWhiteSpace(configuredUrl))
        {
            return configuredUrl.Trim();
        }

        return "ws://127.0.0.1:8001/ws/video";
    }

    private MonoBehaviour FindSharedBackendConfigComponent()
    {
        MonoBehaviour[] behaviours = FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None);
        foreach (MonoBehaviour behaviour in behaviours)
        {
            if (behaviour != null && behaviour.GetType().Name == "BackendConnectionConfig")
            {
                return behaviour;
            }
        }

        return null;
    }

    private string InvokeSharedBackendStringMethod(string methodName)
    {
        if (sharedBackendConfigComponent == null)
        {
            return null;
        }

        var method = sharedBackendConfigComponent.GetType().GetMethod(methodName);
        if (method == null)
        {
            return null;
        }

        object result = method.Invoke(sharedBackendConfigComponent, null);
        return result?.ToString();
    }
}
