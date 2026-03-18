using UnityEngine;

/// <summary>
/// Attached to each face proxy sphere. Manages the material swap between
/// default (not looked at) and highlight (gaze ray hitting it) states,
/// and logs INTERACTION START/END transitions.
///
/// FaceProxyProjector adds this component automatically to spawned proxies when
/// addGazeTargetToSpawnedProxies is enabled; it can also be placed on the prefab directly.
/// FaceProxyGazeInteractor calls SetGazeState() each frame.
/// </summary>
public class FaceProxyGazeTarget : MonoBehaviour
{
    [SerializeField] private Renderer targetRenderer;
    [SerializeField] private Material defaultMaterial;
    [SerializeField] private Material highlightMaterial;

    private bool _isHighlighted;
    private bool _hadInteraction;

    private void Awake()
    {
        if (targetRenderer == null)
        {
            targetRenderer = GetComponentInChildren<Renderer>();
        }
    }

    private void OnEnable()
    {
        ApplyState(false, force: true);
    }

    /// <summary>
    /// Called by FaceProxyProjector after spawning to pass scene-level materials into the proxy.
    /// If either argument is null the existing material assignment is kept.
    /// </summary>
    public void Configure(Material defaultMat, Material highlightMat)
    {
        if (defaultMat != null)
        {
            defaultMaterial = defaultMat;
        }

        if (highlightMat != null)
        {
            highlightMaterial = highlightMat;
        }

        ApplyState(_isHighlighted, force: true);
    }

    /// <summary>
    /// Called by FaceProxyGazeInteractor to notify this target whether the gaze ray is hitting it.
    /// Internally transitions material and logs interaction start/end events.
    /// </summary>
    public void SetGazeState(bool isLookedAt)
    {
        ApplyState(isLookedAt, force: false);
    }

    private void ApplyState(bool isHighlighted, bool force)
    {
        if (!force && _isHighlighted == isHighlighted)
        {
            return;
        }

        _isHighlighted = isHighlighted;

        if (targetRenderer != null)
        {
            Material nextMaterial = _isHighlighted ? highlightMaterial : defaultMaterial;
            if (nextMaterial != null)
            {
                targetRenderer.material = nextMaterial;
            }
        }

        if (force)
        {
            return;
        }

        if (_isHighlighted)
        {
            _hadInteraction = true;
            Debug.Log($"[FaceProxyGazeTarget] INTERACTION START: Gaze ray is now hitting {name}.");
        }
        else if (_hadInteraction)
        {
            Debug.Log($"[FaceProxyGazeTarget] INTERACTION END: Gaze ray left {name}.");
        }
    }
}
