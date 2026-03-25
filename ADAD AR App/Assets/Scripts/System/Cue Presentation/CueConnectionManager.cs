using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Bridge between the AR gaze system and the Python backend cue pipeline.
///
/// Flow:
///   FaceProxyGazeInteractor  →  LSL outlet (AR App)
///   LSL inlet (backend)      →  EEG + face pipeline  →  cue_decision
///   CueConnectionManager     ←  backend (WebSocket push or HTTP polling)
///   CueManager               ←  CueConnectionManager  →  CueDisplay
///
/// Routing modes
/// ─────────────
///   LocalDefault        Simple local testing: spawn a default cue on fixation.
///   LocalCycle          Local testing: cycle through people IDs on each fixation.
///   BackendFaceLookup   HTTP polling for /face/latest, then local cue (legacy).
///   BackendWebSocket    Maintain a persistent WebSocket to /ws/ar.
///                       Backend pushes cue_decision messages whenever the pipeline
///                       produces a result.  OnFixationEvent is used only to record
///                       the current hintTarget for spatial anchoring.
///   BackendEventPolling On every fixation, poll GET /cue/latest until
///                       a decision newer than the fixation timestamp arrives.
///                       Useful when a persistent WebSocket is impractical.
/// </summary>
public class CueConnectionManager : MonoBehaviour
{
    // ── Enums ────────────────────────────────────────────────────────────────────

    public enum CueRoutingMode
    {
        LocalDefault        = 0,
        LocalCycle          = 1,
        BackendFaceLookup   = 2,
        /// <summary>
        /// Maintain a persistent WebSocket to /ws/ar.
        /// The backend pushes a cue_decision message whenever the pipeline produces a result.
        /// OnFixationEvent is captured only for spatial hintTarget anchoring.
        /// </summary>
        BackendWebSocket    = 3,
        /// <summary>
        /// On every fixation, poll GET /cue/latest until a decision newer than the
        /// fixation timestamp arrives or the timeout elapses.
        /// </summary>
        BackendEventPolling = 4,
    }

    // ── DTOs ─────────────────────────────────────────────────────────────────────

    /// Outer envelope for /ws/ar messages: { "type": "cue_decision", "payload": {...} }
    [System.Serializable]
    private class ArWsEnvelopeDto
    {
        public string type;
        public CueDecisionDto payload;
    }

    /// Matches CueDecisionMessage produced by the Python backend.
    [System.Serializable]
    private class CueDecisionDto
    {
        public string event_id;
        public float  event_lsl_timestamp;
        public string face_id;
        public bool   is_unfamiliar;
        public bool   send_cue;
        /// <summary>Populated when send_cue=true. Contains people_id and display params.</summary>
        public CuePayloadDto cue;
        public string server_time;  // UTC ISO-8601
    }

    /// Subset of the cue payload needed on the Unity side.
    /// The rest of the display data (name, photo, audio) is loaded from local Resources
    /// via CueManager.TriggerCueForPerson().
    [System.Serializable]
    private class CuePayloadDto
    {
        public int   people_id;
        public int   font_size_px;
        public float image_scale;
        public float duration_seconds;
    }

    // Legacy DTO — kept for BackendFaceLookup mode only.
    [System.Serializable]
    private class BackendLatestFaceResponseDto
    {
        public string name;
        public int    people_id;
        public float  confidence;
        public string decided_at;
        public string source;
        public float  window_seconds;
        public int    sample_count;
        public bool   is_unknown;
    }

    // ── Serialized fields ─────────────────────────────────────────────────────────

    [Header("References")]
    [SerializeField] private FaceProxyGazeInteractor gazeInteractor;
    [SerializeField] private CueManager cueManager;

    [SerializeField,
     Tooltip("Preferred shared backend address source. Assign the BackendConnectionConfig component here. " +
             "If left empty, the first one found in the scene is used, then legacy URL fallbacks apply.")]
    private MonoBehaviour sharedBackendConfigComponent;

    [Header("Routing")]
    [Tooltip("Select how fixation events map to cue spawning.")]
    [SerializeField] private CueRoutingMode routingMode = CueRoutingMode.BackendWebSocket;

    [Tooltip("When disabled, no routing is triggered regardless of mode.")]
    [SerializeField] private bool spawnCueDirectlyOnFixation = true;

    [Header("Timing")]
    [SerializeField, Min(0f),
     Tooltip("Delay between fixation event and routing logic. Set to 0 for immediate response.")]
    private float fixationToCueDelaySeconds = 0.5f;

    [Header("Local Testing")]
    [Tooltip("Number of people to cycle through when Routing Mode is LocalCycle.")]
    [SerializeField] private int totalPeople = 4;

    [Header("Backend — WebSocket")]
    [Tooltip("Seconds between reconnect attempts when the /ws/ar connection drops.")]
    [SerializeField] private float wsReconnectDelaySeconds = 3f;

    [Tooltip("Receive buffer size in bytes for incoming /ws/ar messages.")]
    [SerializeField] private int wsReceiveBufferBytes = 8192;

    [Header("Backend — Event Polling")]
    [Tooltip("Seconds between GET /cue/latest polls after a fixation event.")]
    [SerializeField] private float pollIntervalSeconds = 0.5f;

    [Tooltip("Give up polling for a decision after this many seconds.")]
    [SerializeField] private float pollTimeoutSeconds = 8f;

    [Tooltip("HTTP request timeout in seconds for each /cue/latest poll attempt.")]
    [SerializeField] private int pollRequestTimeoutSeconds = 3;

    [Header("Backend — Face Lookup (Legacy)")]
    [Tooltip("Lookup configuration for BackendFaceLookup mode.")]
    [SerializeField] private BackendFaceLookupConfig backendLookupConfig = new BackendFaceLookupConfig();

    [Header("Diagnostics")]
    [Tooltip("Log routing decisions and connection events.")]
    [SerializeField] private bool verboseLogs = true;

    // ── Private state ─────────────────────────────────────────────────────────────

    private int  _cycleIndex;
    private bool _backendLookupInFlight;

    // Last known fixation proxy — used as hintTarget for cue anchoring in push/poll modes.
    private Transform _lastKnownHintTarget;

    // WebSocket connection (BackendWebSocket mode)
    private ClientWebSocket _arSocket;
    private CancellationTokenSource _arSocketTokenSource;
    private Coroutine _wsReconnectCoroutine;
    private bool _wsConnectInProgress;

    // Thread-safe queue: async WS receive loop pushes here; Update() drains on the main thread.
    private readonly ConcurrentQueue<CueDecisionDto> _pendingDecisions = new ConcurrentQueue<CueDecisionDto>();

    // Polling (BackendEventPolling mode) — one active session at a time.
    private Coroutine _pollingCoroutine;

    // ── Unity lifecycle ────────────────────────────────────────────────────────────

    private void Awake()
    {
        if (sharedBackendConfigComponent == null)
            sharedBackendConfigComponent = FindSharedBackendConfigComponent();

        if (gazeInteractor == null)
            gazeInteractor = FindFirstObjectByType<FaceProxyGazeInteractor>();

        if (cueManager == null)
            cueManager = FindFirstObjectByType<CueManager>();
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

        if (routingMode == CueRoutingMode.BackendWebSocket)
            StartWsReconnectLoop();
    }

    private void OnDisable()
    {
        if (gazeInteractor != null)
            gazeInteractor.OnFixationEvent -= HandleFixationEvent;

        StopWsReconnectLoop();
        _ = CloseArSocketAsync("Component disabled");

        if (_pollingCoroutine != null)
        {
            StopCoroutine(_pollingCoroutine);
            _pollingCoroutine = null;
        }

        _backendLookupInFlight = false;
    }

    private void Update()
    {
        // Drain decisions queued by the async WebSocket receive loop onto the main thread.
        while (_pendingDecisions.TryDequeue(out CueDecisionDto decision))
            ApplyCueDecision(decision, _lastKnownHintTarget);
    }

    // ── Fixation event handler ─────────────────────────────────────────────────────

    private void HandleFixationEvent(FaceProxyGazeTarget target)
    {
        if (!spawnCueDirectlyOnFixation)
            return;

        if (cueManager == null)
        {
            if (verboseLogs)
                Debug.LogWarning("[CueConnectionManager] CueManager reference missing; cannot spawn cue.");
            return;
        }

        // Always capture the hint target so any incoming decision can anchor correctly.
        _lastKnownHintTarget = target != null ? target.transform : null;

        StartCoroutine(HandleFixationAfterDelay(target, _lastKnownHintTarget));
    }

    private IEnumerator HandleFixationAfterDelay(FaceProxyGazeTarget target, Transform hintTarget)
    {
        float delay = Mathf.Max(0f, fixationToCueDelaySeconds);
        if (delay > 0f)
            yield return new WaitForSeconds(delay);

        switch (routingMode)
        {
            case CueRoutingMode.LocalCycle:
                HandleLocalCycleRoute(target, hintTarget);
                break;

            case CueRoutingMode.BackendFaceLookup:
                if (_backendLookupInFlight)
                {
                    if (verboseLogs)
                        Debug.Log($"[CueConnectionManager] Fixation on {TargetName(target)} ignored: backend lookup already in flight.");
                    yield break;
                }
                StartCoroutine(HandleBackendFaceLookupCoroutine(target, hintTarget));
                break;

            case CueRoutingMode.BackendWebSocket:
                // The LSL outlet in FaceProxyGazeInteractor already sent the fixation marker.
                // hintTarget has been stored in _lastKnownHintTarget above.
                // Nothing more to do here — the cue decision arrives via the WS receive loop.
                if (verboseLogs)
                    Debug.Log($"[CueConnectionManager] Fixation on {TargetName(target)} — hint target captured; awaiting WebSocket cue decision.");
                break;

            case CueRoutingMode.BackendEventPolling:
                // Record when this fixation fired; polling accepts only decisions newer than this.
                DateTime fixationUtc = DateTime.UtcNow;
                if (verboseLogs)
                    Debug.Log($"[CueConnectionManager] Fixation on {TargetName(target)} — starting /cue/latest poll.");
                if (_pollingCoroutine != null)
                    StopCoroutine(_pollingCoroutine);
                _pollingCoroutine = StartCoroutine(PollCueLatestCoroutine(hintTarget, fixationUtc));
                break;

            default: // LocalDefault
                HandleLocalDefaultRoute(target, hintTarget);
                break;
        }
    }

    // ── Backend WebSocket ─────────────────────────────────────────────────────────

    private void StartWsReconnectLoop()
    {
        if (_wsReconnectCoroutine == null)
            _wsReconnectCoroutine = StartCoroutine(WsReconnectLoop());
    }

    private void StopWsReconnectLoop()
    {
        if (_wsReconnectCoroutine != null)
        {
            StopCoroutine(_wsReconnectCoroutine);
            _wsReconnectCoroutine = null;
        }
    }

    private IEnumerator WsReconnectLoop()
    {
        while (enabled && gameObject.activeInHierarchy)
        {
            if (!IsArSocketOpen() && !_wsConnectInProgress)
                _ = ConnectArSocketAsync();

            yield return new WaitForSeconds(wsReconnectDelaySeconds);
        }
    }

    private async Task ConnectArSocketAsync()
    {
        if (_wsConnectInProgress)
            return;

        _wsConnectInProgress = true;
        try
        {
            await CloseArSocketAsync("Reconnect before open");

            _arSocketTokenSource = new CancellationTokenSource();
            _arSocket = new ClientWebSocket();

            Uri wsUri = BuildArWebSocketUri();
            if (verboseLogs)
                Debug.Log($"[CueConnectionManager] Connecting to AR WebSocket: {wsUri}");

            await _arSocket.ConnectAsync(wsUri, _arSocketTokenSource.Token);

            if (verboseLogs)
                Debug.Log("[CueConnectionManager] AR WebSocket connected.");

            _ = ArWebSocketReceiveLoopAsync();
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[CueConnectionManager] AR WebSocket connect failed: {ex.Message}");
        }
        finally
        {
            _wsConnectInProgress = false;
        }
    }

    private async Task ArWebSocketReceiveLoopAsync()
    {
        byte[] buffer = new byte[wsReceiveBufferBytes];
        System.Text.StringBuilder accumulator = new System.Text.StringBuilder();

        try
        {
            while (IsArSocketOpen() && _arSocketTokenSource != null && !_arSocketTokenSource.IsCancellationRequested)
            {
                WebSocketReceiveResult result;
                try
                {
                    result = await _arSocket.ReceiveAsync(
                        new ArraySegment<byte>(buffer),
                        _arSocketTokenSource.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                if (result.MessageType == WebSocketMessageType.Close)
                    break;

                if (result.MessageType != WebSocketMessageType.Text)
                    continue;

                accumulator.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));

                if (!result.EndOfMessage)
                    continue;

                string json = accumulator.ToString();
                accumulator.Clear();

                try
                {
                    ArWsEnvelopeDto envelope = JsonUtility.FromJson<ArWsEnvelopeDto>(json);
                    if (envelope != null && envelope.type == "cue_decision" && envelope.payload != null)
                        _pendingDecisions.Enqueue(envelope.payload);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[CueConnectionManager] Failed to parse WS message: {ex.Message}. " +
                                     $"Raw (first 200): {json.Substring(0, Mathf.Min(200, json.Length))}");
                }
            }
        }
        catch (Exception)
        {
            // Receive loop failures are expected during disconnects/reconnects.
        }
        finally
        {
            await CloseArSocketAsync("Receive loop ended");
        }
    }

    private async Task CloseArSocketAsync(string reason)
    {
        CancellationTokenSource tokenSource = _arSocketTokenSource;
        ClientWebSocket socket = _arSocket;

        _arSocket = null;
        _arSocketTokenSource = null;

        try { tokenSource?.Cancel(); } catch { /* ignored */ }
        try
        {
            if (socket != null && socket.State == WebSocketState.Open)
                await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, reason, CancellationToken.None);
        }
        catch { /* ignored */ }
        finally
        {
            socket?.Dispose();
            tokenSource?.Dispose();
        }
    }

    private bool IsArSocketOpen() =>
        _arSocket != null && _arSocket.State == WebSocketState.Open;

    // ── Backend event polling ──────────────────────────────────────────────────────

    /// <summary>
    /// Polls GET /cue/latest until a decision whose server_time is strictly after
    /// <paramref name="fixationUtc"/> arrives, then passes it to ApplyCueDecision.
    /// Gives up after <see cref="pollTimeoutSeconds"/>.
    /// </summary>
    private IEnumerator PollCueLatestCoroutine(Transform hintTarget, DateTime fixationUtc)
    {
        string url = BuildCueLatestUrl();
        float deadline = Time.realtimeSinceStartup + pollTimeoutSeconds;

        while (Time.realtimeSinceStartup < deadline)
        {
            using (UnityWebRequest req = UnityWebRequest.Get(url))
            {
                req.timeout = pollRequestTimeoutSeconds;
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    CueDecisionDto decision = null;
                    try { decision = JsonUtility.FromJson<CueDecisionDto>(req.downloadHandler.text); }
                    catch (Exception ex)
                    {
                        if (verboseLogs)
                            Debug.LogWarning($"[CueConnectionManager] /cue/latest parse error: {ex.Message}");
                    }

                    if (decision != null && IsDecisionFresh(decision, fixationUtc))
                    {
                        ApplyCueDecision(decision, hintTarget);
                        _pollingCoroutine = null;
                        yield break;
                    }
                }
                else if (req.responseCode != 404 && verboseLogs)
                {
                    Debug.LogWarning($"[CueConnectionManager] /cue/latest: {req.error}");
                }
            }

            yield return new WaitForSeconds(pollIntervalSeconds);
        }

        if (verboseLogs)
            Debug.Log("[CueConnectionManager] Polling timed out; no fresh cue decision received.");

        _pollingCoroutine = null;
    }

    // ── Decision application ────────────────────────────────────────────────────────

    /// <summary>
    /// Final step: act on a backend cue decision on the Unity main thread.
    /// Called from Update() (WebSocket mode) or directly from polling coroutine.
    /// </summary>
    private void ApplyCueDecision(CueDecisionDto decision, Transform hintTarget)
    {
        if (cueManager == null)
        {
            if (verboseLogs)
                Debug.LogWarning("[CueConnectionManager] CueManager missing; cannot apply cue decision.");
            return;
        }

        if (!decision.send_cue)
        {
            if (verboseLogs)
                Debug.Log($"[CueConnectionManager] Backend decision (event={decision.event_id}): " +
                          $"send_cue=false (is_unfamiliar={decision.is_unfamiliar}). No cue spawned.");
            return;
        }

        if (decision.cue == null || decision.cue.people_id <= 0)
        {
            if (verboseLogs)
                Debug.LogWarning($"[CueConnectionManager] Backend decision (event={decision.event_id}): " +
                                 "send_cue=true but invalid/missing cue payload.");
            return;
        }

        if (cueManager.CurrentPeopleId == decision.cue.people_id)
        {
            if (verboseLogs)
                Debug.Log($"[CueConnectionManager] people_id={decision.cue.people_id} already on screen; skipping re-spawn.");
            return;
        }

        bool started = cueManager.TriggerCueForPerson(decision.cue.people_id, hintTarget);
        if (verboseLogs)
        {
            Debug.Log(started
                ? $"[CueConnectionManager] Backend decision → people_id={decision.cue.people_id} cue started (event={decision.event_id})."
                : $"[CueConnectionManager] Backend decision → TriggerCueForPerson({decision.cue.people_id}) not started.");
        }
    }

    private static bool IsDecisionFresh(CueDecisionDto decision, DateTime fixationUtc)
    {
        if (string.IsNullOrEmpty(decision.server_time))
            return false;

        if (!DateTime.TryParse(decision.server_time, null,
            System.Globalization.DateTimeStyles.RoundtripKind, out DateTime serverTime))
            return false;

        return serverTime.ToUniversalTime() > fixationUtc;
    }

    // ── Local test routes ──────────────────────────────────────────────────────────

    private void HandleLocalCycleRoute(FaceProxyGazeTarget target, Transform hintTarget)
    {
        int count = Mathf.Max(1, totalPeople);
        int nextPeopleId = (_cycleIndex % count) + 1;

        if (cueManager.CurrentPeopleId == nextPeopleId)
            nextPeopleId = (nextPeopleId % count) + 1;

        bool started = cueManager.TriggerCueForPerson(nextPeopleId, hintTarget);
        if (started) _cycleIndex = nextPeopleId;

        if (verboseLogs)
        {
            Debug.Log(started
                ? $"[CueConnectionManager] Fixation on {TargetName(target)} -> cycling to people_id={nextPeopleId}/{count}."
                : $"[CueConnectionManager] Fixation on {TargetName(target)} -> TriggerCueForPerson({nextPeopleId}) not started.");
        }
    }

    private void HandleLocalDefaultRoute(FaceProxyGazeTarget target, Transform hintTarget)
    {
        if (cueManager.HasActiveCue)
        {
            if (verboseLogs)
                Debug.Log($"[CueConnectionManager] Fixation on {TargetName(target)}, but cue already active. Skipping spawn.");
            return;
        }

        bool started = cueManager.TriggerCueIfNone(hintTarget);
        if (verboseLogs)
        {
            Debug.Log(started
                ? $"[CueConnectionManager] Fixation on {TargetName(target)} -> requested cue spawn."
                : $"[CueConnectionManager] Fixation on {TargetName(target)} -> spawn request not started.");
        }
    }

    // ── Legacy: BackendFaceLookup ──────────────────────────────────────────────────

    private IEnumerator HandleBackendFaceLookupCoroutine(FaceProxyGazeTarget target, Transform hintTarget)
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
                        Debug.LogWarning($"[CueConnectionManager] Face lookup failed ({url}): {req.error}");
                    yield break;
                }

                string json = req.downloadHandler.text;
                bool peopleIdWasNull = json.Contains("\"people_id\":null");

                BackendLatestFaceResponseDto latest = JsonUtility.FromJson<BackendLatestFaceResponseDto>(json);
                if (latest == null)
                {
                    if (verboseLogs)
                        Debug.LogWarning("[CueConnectionManager] Face lookup parse failed: null response DTO.");
                    yield break;
                }

                if (peopleIdWasNull)
                {
                    if (verboseLogs)
                        Debug.Log($"[CueConnectionManager] Fixation on {TargetName(target)} -> no-face (people_id=null).");
                    yield break;
                }

                if (latest.is_unknown || latest.people_id == 0)
                {
                    if (verboseLogs)
                        Debug.Log($"[CueConnectionManager] Fixation on {TargetName(target)} -> Unknown person. Skipping cue.");
                    yield break;
                }

                if (latest.people_id <= 0)
                {
                    if (verboseLogs)
                        Debug.LogWarning($"[CueConnectionManager] Fixation on {TargetName(target)} -> invalid people_id={latest.people_id}.");
                    yield break;
                }

                if (cueManager.CurrentPeopleId == latest.people_id)
                {
                    if (verboseLogs)
                        Debug.Log($"[CueConnectionManager] people_id={latest.people_id} already active. No respawn.");
                    yield break;
                }

                bool started = cueManager.TriggerCueForPerson(latest.people_id, hintTarget);
                if (verboseLogs)
                {
                    Debug.Log(started
                        ? $"[CueConnectionManager] Face lookup -> people_id={latest.people_id} confidence={latest.confidence:F2}, cue started."
                        : $"[CueConnectionManager] TriggerCueForPerson({latest.people_id}) not started.");
                }
            }
        }
        finally
        {
            _backendLookupInFlight = false;
        }
    }

    // ── URL builders ──────────────────────────────────────────────────────────────

    /// <summary>WebSocket URI for the AR cue push channel (/ws/ar).</summary>
    public Uri BuildArWebSocketUri()
    {
        var method = sharedBackendConfigComponent?.GetType().GetMethod("BuildArWebSocketUri");
        if (method != null && method.Invoke(sharedBackendConfigComponent, null) is Uri uri)
            return uri;

        // Derive from video WS URL if available, otherwise use hardcoded default.
        string baseUrl = backendLookupConfig?.videoWsUrl;
        if (!string.IsNullOrWhiteSpace(baseUrl))
        {
            UriBuilder builder = new UriBuilder(baseUrl.Trim()) { Path = "/ws/ar" };
            return builder.Uri;
        }

        return new Uri("ws://127.0.0.1:8001/ws/ar");
    }

    /// <summary>HTTP URL for the GET /cue/latest polling endpoint.</summary>
    public string BuildCueLatestUrl()
    {
        var method = sharedBackendConfigComponent?.GetType().GetMethod("BuildCueLatestUrl");
        if (method != null && method.Invoke(sharedBackendConfigComponent, null) is string s
            && !string.IsNullOrWhiteSpace(s))
            return s.Trim();

        string baseUrl = backendLookupConfig?.faceLookupUrl;
        if (!string.IsNullOrWhiteSpace(baseUrl))
        {
            UriBuilder builder = new UriBuilder(baseUrl.Trim()) { Path = "/cue/latest" };
            return builder.Uri.ToString();
        }

        return "http://127.0.0.1:8001/cue/latest";
    }

    private string BuildLatestFaceUrl()
    {
        string sharedUrl = InvokeSharedBackendStringMethod("BuildFaceLookupUrl");
        if (!string.IsNullOrWhiteSpace(sharedUrl)) return sharedUrl.Trim();

        string configuredUrl = backendLookupConfig?.faceLookupUrl;
        if (!string.IsNullOrWhiteSpace(configuredUrl)) return configuredUrl.Trim();

        return "http://127.0.0.1:8001/face/latest";
    }

    /// <summary>
    /// HTTP→WS URL for the video stream. Read by FaceVideoStreamSender which looks
    /// this up on the CueConnectionManager as a shared config source.
    /// </summary>
    public string BuildVideoWebSocketUrl()
    {
        string sharedUrl = InvokeSharedBackendStringMethod("BuildVideoWebSocketUri");
        if (!string.IsNullOrWhiteSpace(sharedUrl)) return sharedUrl.Trim();

        string configuredUrl = backendLookupConfig?.videoWsUrl;
        if (!string.IsNullOrWhiteSpace(configuredUrl)) return configuredUrl.Trim();

        return "ws://127.0.0.1:8001/ws/video";
    }

    // ── Shared config helpers ──────────────────────────────────────────────────────

    private MonoBehaviour FindSharedBackendConfigComponent()
    {
        MonoBehaviour[] behaviours = FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None);
        foreach (MonoBehaviour behaviour in behaviours)
        {
            if (behaviour != null && behaviour.GetType().Name == "BackendConnectionConfig")
                return behaviour;
        }
        return null;
    }

    private string InvokeSharedBackendStringMethod(string methodName)
    {
        if (sharedBackendConfigComponent == null) return null;
        var method = sharedBackendConfigComponent.GetType().GetMethod(methodName);
        if (method == null) return null;
        return method.Invoke(sharedBackendConfigComponent, null)?.ToString();
    }

    private static string TargetName(FaceProxyGazeTarget target) =>
        target != null ? target.name : "null";
}

