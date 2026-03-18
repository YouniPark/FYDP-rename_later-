using UnityEngine;

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
    [Header("References")]
    [SerializeField] private FaceProxyGazeInteractor gazeInteractor;
    [SerializeField] private CueManager cueManager;

    [Header("Routing")]
    [Tooltip("When enabled, fixation events directly trigger a default cue spawn request.")]
    [SerializeField] private bool spawnCueDirectlyOnFixation = true;

    [Tooltip("When enabled, each fixation cycles to the next person ID (1..totalPeople) instead of using the default cue.")]
    [SerializeField] private bool cycleOnFixation = false;

    [Tooltip("Number of people to cycle through when cycleOnFixation is enabled.")]
    [SerializeField] private int totalPeople = 4;

    [Tooltip("Log routing decisions for debugging while wiring connections.")]
    [SerializeField] private bool verboseLogs = true;

    private int _cycleIndex = 0;

    private void Awake()
    {
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

        if (cycleOnFixation)
        {
            // Advance to the next person and present a new cue only if its people_id differs
            // from the cue currently on screen. This mirrors the server-side face-id change logic.
            int count = Mathf.Max(1, totalPeople);
            int nextPeopleId = (_cycleIndex % count) + 1; // cycles 1 → 2 → ... → totalPeople → 1

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
            return;
        }

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
}
