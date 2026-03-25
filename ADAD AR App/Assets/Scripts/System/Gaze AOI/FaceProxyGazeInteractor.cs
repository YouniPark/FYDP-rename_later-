using LSL;
using MagicLeap.OpenXR.Features.EyeTracker;
using UnityEngine;

/// <summary>
/// Casts the EyeGazeRayProvider's gaze ray each frame and applies hover state to whichever
/// FaceProxyGazeTarget collider it hits. Only one proxy can be active at a time — transitioning
/// to a new target automatically de-activates the previous one.
///
/// Also tracks the ML gaze behavior type (Fixation, Saccade, Pursuit, etc.) while a proxy
/// is under gaze. Behavior-type changes are logged for calibration, and a special FIXATION EVENT
/// is logged the first time a fixation-type behavior is observed during a continuous
/// proxy-collision window (regardless of which proxy is hit).
/// </summary>
public class FaceProxyGazeInteractor : MonoBehaviour
{
    [SerializeField] private EyeGazeRayProvider gazeProvider;

    /// <summary>
    /// Fired when the first fixation is detected during the current continuous proxy-collision window.
    /// Payload is the currently hit FaceProxyGazeTarget (may be null in edge cases).
    /// </summary>
    public event System.Action<FaceProxyGazeTarget> OnFixationEvent;

    [Tooltip("Maximum distance for gaze raycast hits.")]
    [SerializeField] private float maxRayDistance = 10f;

    [Tooltip("Layer mask for gaze raycast. Should include the layer(s) where FaceProxy colliders are.")]
    [SerializeField] private LayerMask raycastLayerMask = ~0;

    [Tooltip("If enabled, a debug ray will be drawn in the Scene view showing the gaze ray each frame.")]
    [SerializeField] private bool drawDebugRay = false;

    [Tooltip("If enabled, logs gaze behavior transitions while targeting a proxy.")]
    [SerializeField] private bool logGazeBehavior = false;

    [Tooltip("When true, proxy highlight starts only after first fixation event in a collision window. When false, highlight starts on collision.")]
    [SerializeField] private bool highlightOnFixation = true;

    // LSL outlet for fixation event markers
    private StreamOutlet _lslOutlet;

    // The proxy collider currently under gaze (null if gaze hits nothing)
    private FaceProxyGazeTarget _currentTarget;

    // Behavior tracking state during continuous proxy collision
    private string _lastBehaviorName;
    private bool _fixationLoggedForCollision;
    private bool _wasCollidingLastFrame;

    private void Awake()
    {
        // Fallback: find the provider in the scene if it wasn't assigned in the Inspector
        if (gazeProvider == null)
        {
            gazeProvider = FindFirstObjectByType<EyeGazeRayProvider>();
        }

        // Create the LSL outlet for fixation event markers.
        // One string channel, irregular rate — each push is a single marker.
        var streamInfo = new StreamInfo(
            name: "FixationEvents",
            type: "Markers",
            channel_count: 1,
            nominal_srate: LSL.LSL.IRREGULAR_RATE,
            channel_format: channel_format_t.cf_string,
            source_id: "FaceProxyGazeInteractor"
        );
        _lslOutlet = new StreamOutlet(streamInfo);
    }

    private void Update()
    {
        // If no valid gaze ray is available this frame, release the active target
        if (gazeProvider == null || !gazeProvider.TryGetGazeRay(out Ray gazeRay))
        {
            ClearCurrentTarget();
            return;
        }

        if (drawDebugRay)
        {
            Debug.DrawRay(gazeRay.origin, gazeRay.direction * maxRayDistance, Color.green);
        }

        // Raycast along the gaze ray; look for a FaceProxyGazeTarget on or above the hit object
        FaceProxyGazeTarget hitTarget = null;
        if (Physics.Raycast(gazeRay, out RaycastHit hitInfo, maxRayDistance, raycastLayerMask, QueryTriggerInteraction.Collide))
        {
            hitTarget = hitInfo.collider.GetComponentInParent<FaceProxyGazeTarget>();
        }

        // On target change: notify old target, update current, reset behavior state, notify new target
        if (hitTarget != _currentTarget)
        {
            if (_currentTarget != null)
            {
                _currentTarget.SetGazeState(false);
            }

            _currentTarget = hitTarget;

            if (_currentTarget != null)
            {
                _currentTarget.SetGazeState(ShouldHighlightCurrentTarget());
            }
        }

        // While a proxy is active, track gaze behavior for fixation detection
        if (_currentTarget != null)
        {
            TrackBehaviorWhileOnTarget();
        }
        else if (_wasCollidingLastFrame)
        {
            // Collision ended: reset per-collision state for next engagement
            _lastBehaviorName = null;
            _fixationLoggedForCollision = false;
        }

        _wasCollidingLastFrame = _currentTarget != null;
    }

    /// <summary>
    /// Polls EyeGazeRayProvider's gaze behavior while a proxy is being looked at.
    /// Logs behavior-type transitions (useful for discovering enum names on device).
    /// Fires a one-time FIXATION EVENT when fixation is first detected during the current
    /// continuous proxy-collision window.
    /// </summary>
    private void TrackBehaviorWhileOnTarget()
    {
        if (!gazeProvider.TryGetGazeBehavior(out GazeBehavior behavior)) return;

        // ToString() returns the enum member name (e.g. "Fixation", "Saccade").
        // Check the device log after first run to confirm the exact names for this SDK version.
        string behaviorName = behavior.GazeBehaviorType.ToString();

        // Log only on transition to avoid per-frame spam
        if (behaviorName != _lastBehaviorName)
        {
            if (logGazeBehavior)
            {
                Debug.Log($"[FaceProxyGazeInteractor] Gaze behavior: '{behaviorName}' while targeting {_currentTarget.name}.");
            }

            _lastBehaviorName = behaviorName;
        }

        // Fire a special one-time log the first time fixation is detected on this target.
        if (!_fixationLoggedForCollision && behaviorName.ToLower().Contains("fixation"))
        {
            Debug.Log($"[FaceProxyGazeInteractor] FIXATION EVENT: First fixation detected while gaze is colliding with a face proxy (current={_currentTarget.name}).");
            _fixationLoggedForCollision = true;

            // Push an LSL event marker containing the face proxy's name.
            _lslOutlet?.push_sample(new string[] { _currentTarget.name });

            OnFixationEvent?.Invoke(_currentTarget);

            if (_currentTarget != null && highlightOnFixation)
            {
                _currentTarget.SetGazeState(true);
            }
        }
    }

    private bool ShouldHighlightCurrentTarget()
    {
        return _currentTarget != null && (!highlightOnFixation || _fixationLoggedForCollision);
    }

    /// <summary>
    /// Allows FaceGazeSystemMain to inject the gaze provider at startup before all serialized
    /// references would otherwise resolve, ensuring the ray source is always set.
    /// </summary>
    public void SetGazeProvider(EyeGazeRayProvider provider)
    {
        gazeProvider = provider;
    }

    private void OnDestroy()
    {
        _lslOutlet?.Dispose();
        _lslOutlet = null;
    }

    private void OnDisable()
    {
        ClearCurrentTarget();
    }

    private void ClearCurrentTarget()
    {
        if (_currentTarget != null)
        {
            _currentTarget.SetGazeState(false);
            _currentTarget = null;
        }
    }
}
