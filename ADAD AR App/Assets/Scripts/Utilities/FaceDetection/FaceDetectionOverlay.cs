using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Draws bounding boxes over a RawImage to visualise BlazeFace detections.
///
/// Scene setup (see Inspector tooltips):
///   - displayImage  : the RawImage showing the camera feed
///   - detector      : the BlazeFaceDetector component
///   - boxPrefab     : a UI prefab with a RectTransform + Image (use a border/outline sprite)
///   - overlayCanvas : the Canvas that sits in front of displayImage (same parent or sibling)
/// </summary>
public class FaceDetectionOverlay : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private RawImage          displayImage;
    [SerializeField] private BlazeFaceDetector detector;

    [Header("Box Prefab")]
    [Tooltip("UI prefab with a RectTransform and Image. Use a transparent-fill sprite with a coloured border.")]
    [SerializeField] private RectTransform boxPrefab;

    [Tooltip("Parent RectTransform that the box instances are spawned under. Should cover the same area as the RawImage.")]
    [SerializeField] private RectTransform overlayRoot;

    private readonly List<RectTransform> _boxPool = new List<RectTransform>();
    private int _activeFaces;
    private int _lastLoggedFaceCount = -1;

    private void OnEnable()
    {
        if (detector != null)
            detector.OnFacesDetected += OnFacesDetected;
    }

    private void OnDisable()
    {
        if (detector != null)
            detector.OnFacesDetected -= OnFacesDetected;

        HideAll();
    }

    private void OnFacesDetected(BlazeFaceDetector.DetectedFace[] faces)
    {
        bool countChanged = faces.Length != _lastLoggedFaceCount;

        if (countChanged)
        {
            Debug.Log($"[FaceOverlay] Face count changed to {faces.Length}. " +
                      $"displayImage={(displayImage != null ? "assigned" : "NULL")}, " +
                      $"overlayRoot={(overlayRoot != null ? "assigned" : "NULL")}");
            _lastLoggedFaceCount = faces.Length;
        }
        _activeFaces = faces.Length;

        // Grow pool on demand.
        while (_boxPool.Count < faces.Length)
        {
            var instance = Instantiate(boxPrefab, overlayRoot);
            instance.gameObject.SetActive(false);
            _boxPool.Add(instance);
        }

        // Hide boxes beyond detection count.
        for (int i = faces.Length; i < _boxPool.Count; i++)
            _boxPool[i].gameObject.SetActive(false);

        if (displayImage == null || faces.Length == 0) return;

        // The RawImage's RectTransform gives us the pixel rect in local space.
        Rect imageRect = displayImage.rectTransform.rect;

        for (int i = 0; i < faces.Length; i++)
        {
            var box = _boxPool[i];
            box.gameObject.SetActive(true);

            Rect norm = faces[i].boundingBox;

            // Convert normalized [0,1] face coords to pixel size/position inside imageRect.
            // No Y-flip needed: the detector's affine transform already handles the frame orientation.
            // norm.x and norm.y are the top-left corner in normalized image space.
            float pixX = imageRect.xMin + norm.x      * imageRect.width;
            float pixY = imageRect.yMin + norm.y      * imageRect.height;
            float pixW = norm.width  * imageRect.width;
            float pixH = norm.height * imageRect.height;

            if (i == 0 && (pixW < 2f || pixH < 2f) && countChanged)
            {
                Debug.LogWarning($"[FaceOverlay] First box is tiny ({pixW:F1}x{pixH:F1}px). " +
                                 "Detections may be valid but coordinates/scaling likely mismatch.");
            }

            // anchoredPosition is the centre of the box (pivot assumed 0.5, 0.5).
            box.anchoredPosition = new Vector2(pixX + pixW * 0.5f, pixY + pixH * 0.5f);
            box.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, pixW);
            box.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical,   pixH);
        }
    }

    private void HideAll()
    {
        foreach (var box in _boxPool)
            if (box != null) box.gameObject.SetActive(false);
    }
}
