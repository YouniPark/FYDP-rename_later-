using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class CueDisplay : MonoBehaviour
{
    [Header("Target")]
    [Tooltip("assuming that the head/face is a 3d object in unity")]
    public Transform target;

    [Header("Placement")]
    [Tooltip("distance btw head and cue")]
    public Vector3 cardOffset = new Vector3(-0.25f, 0.15f, 0f);

    [Header("Text")]
    public string personName = "tartaglia ajax";
    public string relationship = "friend";

    [Header("Image")]
    [Tooltip("photo below cue text")]
    public Sprite personPhoto;

    [Header("Style")]
    public Vector2 panelSize = new Vector2(0.34f, 0.22f);
    public Color panelColor = Color.white;
    public Color textColor = Color.black;
    public Color lineColor = Color.white;
    [Range(0.001f, 0.02f)] public float lineWidth = 0.004f;

    private Transform _cardRoot;
    private RectTransform _panelRect;
    private LineRenderer _line;
    private Camera _viewCamera;

    private void Start()
    {
        if (target == null)
        {
            Debug.LogError("no head available");
            enabled = false;
            return;
        }

        _viewCamera = Camera.main;

        BuildCueCard();
        BuildConnectorLine();
    }

    private void LateUpdate()
    {
        if (_cardRoot == null || target == null)
            return;

        // position
        _cardRoot.position = target.position + cardOffset;

        // cue face camera view
        if (_viewCamera == null)
            _viewCamera = Camera.main;

        if (_viewCamera != null)
        {
            Vector3 forward = (_cardRoot.position - _viewCamera.transform.position).normalized;
            _cardRoot.rotation = Quaternion.LookRotation(forward, Vector3.up);
        }

        UpdateConnectorLine();
    }

    private void BuildCueCard()
    {
        GameObject root = new GameObject("CueCardRoot");
        _cardRoot = root.transform;

        var canvas = root.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.worldCamera = _viewCamera;

        var scaler = root.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 1000f;
        root.AddComponent<GraphicRaycaster>();

        GameObject panelGO = new GameObject("Panel", typeof(RectTransform), typeof(Image));
        panelGO.transform.SetParent(root.transform, false);
        _panelRect = panelGO.GetComponent<RectTransform>();
        _panelRect.sizeDelta = panelSize * 1000f; // 1 unit about 1 m in unity --> NEED TO TEST?

        var panelImage = panelGO.GetComponent<Image>();
        panelImage.color = panelColor;

        // text
        GameObject textGO = new GameObject("InfoText", typeof(RectTransform), typeof(TextMeshProUGUI));
        textGO.transform.SetParent(panelGO.transform, false);

        RectTransform textRect = textGO.GetComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0f, 0.48f);
        textRect.anchorMax = new Vector2(1f, 1f);
        textRect.offsetMin = new Vector2(20f, 10f);
        textRect.offsetMax = new Vector2(-20f, -10f);

        var tmp = textGO.GetComponent<TextMeshProUGUI>();
        tmp.text = $"Name: {personName}\nRelationship: {relationship}";
        tmp.fontSize = 42;
        tmp.color = textColor;
        tmp.alignment = TextAlignmentOptions.MidlineLeft;

        // image
        GameObject photoGO = new GameObject("Photo", typeof(RectTransform), typeof(Image));
        photoGO.transform.SetParent(panelGO.transform, false);

        RectTransform photoRect = photoGO.GetComponent<RectTransform>();
        photoRect.anchorMin = new Vector2(0f, 0f);
        photoRect.anchorMax = new Vector2(1f, 0.48f);
        photoRect.offsetMin = new Vector2(0f, 0f);
        photoRect.offsetMax = new Vector2(0f, 0f);

        var photoImage = photoGO.GetComponent<Image>();
        photoImage.sprite = personPhoto;
        photoImage.preserveAspect = true;
        photoImage.color = personPhoto == null ? new Color(0.85f, 0.85f, 0.85f, 1f) : Color.white;
    }

    private void BuildConnectorLine()
    {
        GameObject lineGO = new GameObject("CueConnectorLine");
        _line = lineGO.AddComponent<LineRenderer>();
        _line.positionCount = 2;
        _line.useWorldSpace = true;
        _line.startWidth = lineWidth;
        _line.endWidth = lineWidth;
        _line.material = new Material(Shader.Find("Sprites/Default"));
        _line.startColor = lineColor;
        _line.endColor = lineColor;

        UpdateConnectorLine();
    }

    private void UpdateConnectorLine() // connects cue to face
    {
        if (_line == null || _panelRect == null || target == null)
            return;

        Vector3 panelStart = _panelRect.TransformPoint(new Vector3(_panelRect.rect.xMax, 0f, 0f));
        Vector3 targetPoint = target.position;

        _line.SetPosition(0, panelStart);
        _line.SetPosition(1, targetPoint);
    }
}