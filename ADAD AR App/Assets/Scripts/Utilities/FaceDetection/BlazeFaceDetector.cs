using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using Unity.InferenceEngine;
using Unity.Mathematics;

/// <summary>
/// Runs BlazeFace detection on frames coming from ImageStream.
/// Requires a BlazeFace model that already includes NMS in the graph.
/// </summary>
public class BlazeFaceDetector : MonoBehaviour
{
    [Header("Model")]
    [SerializeField] private ModelAsset blazeFaceModelAsset;
    [SerializeField] private TextAsset anchorsCsv;

    [Header("Camera Source")]
    [SerializeField] private ImageStream cameraSource;

    public ImageStream CameraSource => cameraSource;

    [Header("Detection Settings")]
    public float scoreThreshold = 0.5f;
    public float iouThreshold = 0.3f;
    [Tooltip("Hard cap on faces emitted each tick after score filtering.")]
    public int maxFacesToEmit = 5;
    [Tooltip("Reject tiny boxes that are usually noise (normalized size in 0..1 space).")]
    public float minBoxSizeNormalized = 0.05f;

    [Header("Debug")]
    [Tooltip("Enable/disable repetitive BlazeFace face-count threshold logs.")]
    [SerializeField] private bool logDetectionThresholdMessages = false;

    [Header("Inference Rate")]
    public float detectionHz = 10f;

    public struct DetectedFace
    {
        public Rect boundingBox;
        public Vector2 center;
        public float confidence;
    }

    public DetectedFace[] DetectedFaces { get; private set; } = Array.Empty<DetectedFace>();
    public event Action<DetectedFace[]> OnFacesDetected;

    private Model runtimeModel;
    private Worker worker;

    private Tensor<float> inputTensor;

    private const int InputSize = 128;

    private float nextDetectTime;
    private bool isBusy;
    private int _lastLoggedFaceCount = -1;
    private int _nullFrameSkipCount;
    private bool _loggedShapeInfo;
    private bool _usingCompiledNmsModel;
    private float2x3 _tensorToImage;
    private float _currentFrameWidth;
    private float _currentFrameHeight;

    private float[,] _anchors;
    private const int NumAnchorValues = 4;

    void Start()
    {
        InitializeModel();
    }

    void Update()
    {
        if (worker == null || isBusy) return;
        if (Time.time < nextDetectTime) return;

        // Advance the timer regardless so null-frame counting stays in sync with detection rate.
        nextDetectTime = Time.time + 1f / Mathf.Max(1f, detectionHz);

        Texture frame = cameraSource != null ? cameraSource.GetLatestFrameTexture() : null;
        if (frame == null)
        {
            _nullFrameSkipCount++;
            // Log on first occurrence and every 10 ticks after to avoid spam.
            if (_nullFrameSkipCount == 1 || _nullFrameSkipCount % 10 == 0)
                Debug.LogWarning($"[BlazeFace] Frame is null (tick #{_nullFrameSkipCount}). " +
                                  "Camera provider may still be warming up or capture has not started yet.");
            return;
        }

        _nullFrameSkipCount = 0;
        DetectFaces(frame);
    }

    void InitializeModel()
    {
        if (blazeFaceModelAsset == null)
        {
            Debug.LogError("BlazeFace model not assigned.");
            return;
        }

        var baseModel = ModelLoader.Load(blazeFaceModelAsset);

        _anchors = TryLoadAnchors(anchorsCsv);
        if (_anchors != null)
        {
            try
            {
                var numAnchors = _anchors.GetLength(0);
                var graph = new FunctionalGraph();
                var input = graph.AddInput(baseModel, 0);
                var outputs = Functional.Forward(baseModel, 2 * input - 1);

                if (outputs.Length < 2)
                    throw new InvalidOperationException("Model did not return expected boxes/scores outputs.");

                var boxes = outputs[0];
                var scores = outputs[1];

                var anchorsData = new float[numAnchors * NumAnchorValues];
                Buffer.BlockCopy(_anchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
                var anchorsTensor = Functional.Constant(new TensorShape(numAnchors, NumAnchorValues), anchorsData);

                var filtered = BlazeUtils.NMSFiltering(boxes, scores, anchorsTensor, InputSize, iouThreshold, scoreThreshold);
                runtimeModel = graph.Compile(filtered.Item1, filtered.Item2, filtered.Item3);
                _usingCompiledNmsModel = true;
                Debug.Log($"BlazeFaceDetector initialized with anchor+NMS graph ({numAnchors} anchors).");
            }
            catch (Exception ex)
            {
                runtimeModel = baseModel;
                _usingCompiledNmsModel = false;
                Debug.LogWarning("[BlazeFace] Failed to compile anchor+NMS postprocess graph, using raw outputs: " + ex.Message);
            }
        }
        else
        {
            runtimeModel = baseModel;
            _usingCompiledNmsModel = false;
            Debug.LogWarning("[BlazeFace] No anchors CSV assigned. Using fallback raw-output decoding; detections may be noisy.");
        }

        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        inputTensor = new Tensor<float>(new TensorShape(1, InputSize, InputSize, 3));

        Debug.Log("BlazeFaceDetector initialized");
    }

    public void DetectFaces(Texture source)
    {
        if (worker == null || source == null || isBusy) return;

        isBusy = true;

        try
        {
            _currentFrameWidth = source.width;
            _currentFrameHeight = source.height;

            // Match the sample preprocessing: square letterbox transform from 128x128 tensor to image space.
            var size = Mathf.Max(source.width, source.height);
            var scale = size / (float)InputSize;
            _tensorToImage = BlazeUtils.mul(
                BlazeUtils.TranslationMatrix(0.5f * (new float2(source.width, source.height) + new float2(-size, size))),
                BlazeUtils.ScaleMatrix(new float2(scale, -scale))
            );

            BlazeUtils.SampleImageAffine(source, inputTensor, _tensorToImage);

            worker.Schedule(inputTensor);

            var faces = _usingCompiledNmsModel ? DecodeCompiledOutputs() : DecodeRawOutputs();
            int faceCount = faces.Length;

            if (logDetectionThresholdMessages && faceCount != _lastLoggedFaceCount)
            {
                if (faceCount > 0)
                    Debug.Log($"[BlazeFace] {faceCount} face(s) detected. Confidences: " +
                              string.Join(", ", System.Array.ConvertAll(faces, f => f.confidence.ToString("F2"))));
                else
                    Debug.Log($"[BlazeFace] No faces passed threshold ({scoreThreshold:F2}). Mode={(_usingCompiledNmsModel ? "NMS" : "raw")}");
                _lastLoggedFaceCount = faceCount;
            }

            DetectedFaces = faces;
            OnFacesDetected?.Invoke(faces);
        }
        catch (Exception ex)
        {
            Debug.LogError("BlazeFace inference error: " + ex.Message);
            DetectedFaces = Array.Empty<DetectedFace>();
        }
        finally
        {
            isBusy = false;
        }
    }

    private DetectedFace[] DecodeCompiledOutputs()
    {
        var indices = worker.PeekOutput(0) as Tensor<int>;
        var scores = worker.PeekOutput(1) as Tensor<float>;
        var boxes = worker.PeekOutput(2) as Tensor<float>;

        if (indices == null || scores == null || boxes == null || _anchors == null)
            return Array.Empty<DetectedFace>();

        using var indicesReadable = indices.ReadbackAndClone();
        using var scoresReadable = scores.ReadbackAndClone();
        using var boxesReadable = boxes.ReadbackAndClone();

        if (!_loggedShapeInfo)
        {
            Debug.Log($"[BlazeFace] Output shapes -> indices: {indicesReadable.shape}, scores: {scoresReadable.shape}, boxes: {boxesReadable.shape}");
            _loggedShapeInfo = true;
        }

        int selectedCount = indicesReadable.shape.length;
        int anchorCount = _anchors.GetLength(0);
        var faceList = new List<DetectedFace>(Mathf.Min(selectedCount, maxFacesToEmit));

        for (int i = 0; i < selectedCount; i++)
        {
            int idx = indicesReadable[i];
            if (idx < 0 || idx >= anchorCount)
                continue;

            float confidence = Sigmoid(scoresReadable[0, i, 0]);

            float2 anchorPosition = InputSize * new float2(_anchors[idx, 0], _anchors[idx, 1]);
            float2 boxCenterImage = BlazeUtils.mul(_tensorToImage, anchorPosition + new float2(boxesReadable[0, i, 0], boxesReadable[0, i, 1]));
            float2 boxTopRightImage = BlazeUtils.mul(_tensorToImage, anchorPosition + new float2(
                boxesReadable[0, i, 0] + 0.5f * boxesReadable[0, i, 2],
                boxesReadable[0, i, 1] + 0.5f * boxesReadable[0, i, 3]));
            float2 boxSizeImage = 2f * (boxTopRightImage - boxCenterImage);

            float normCx = boxCenterImage.x / Mathf.Max(1f, _currentFrameWidth);
            float normCy = boxCenterImage.y / Mathf.Max(1f, _currentFrameHeight);
            float normW = math.abs(boxSizeImage.x) / Mathf.Max(1f, _currentFrameWidth);
            float normH = math.abs(boxSizeImage.y) / Mathf.Max(1f, _currentFrameHeight);

            if (!TryBuildFace(normCx, normCy, normW, normH, confidence, out var face))
                continue;

            faceList.Add(face);
        }

        if (faceList.Count > maxFacesToEmit)
            faceList.RemoveRange(maxFacesToEmit, faceList.Count - maxFacesToEmit);

        return faceList.ToArray();
    }

    private DetectedFace[] DecodeRawOutputs()
    {
        var boxes = worker.PeekOutput("boxes") as Tensor<float>;
        var scores = worker.PeekOutput("scores") as Tensor<float>;

        if (boxes == null || scores == null)
            return Array.Empty<DetectedFace>();

        using var boxesReadable = boxes.ReadbackAndClone();
        using var scoresReadable = scores.ReadbackAndClone();

        if (!_loggedShapeInfo)
        {
            Debug.Log($"[BlazeFace] Output shapes -> boxes: {boxesReadable.shape}, scores: {scoresReadable.shape}");
            _loggedShapeInfo = true;
        }

        int numBoxes = boxesReadable.shape[1];
        var faceList = new List<DetectedFace>(Mathf.Min(numBoxes, maxFacesToEmit));

        for (int i = 0; i < numBoxes; i++)
        {
            float confidence = Sigmoid(scoresReadable[0, i]);
            if (confidence < scoreThreshold)
                continue;

            float cx = boxesReadable[0, i, 0];
            float cy = boxesReadable[0, i, 1];
            float w = boxesReadable[0, i, 2];
            float h = boxesReadable[0, i, 3];

            bool appearsNormalized = Mathf.Abs(cx) <= 1.5f && Mathf.Abs(cy) <= 1.5f && Mathf.Abs(w) <= 1.5f && Mathf.Abs(h) <= 1.5f;

            float normCx;
            float normCy;
            float normW;
            float normH;

            if (appearsNormalized)
            {
                normCx = cx;
                normCy = cy;
                normW = w;
                normH = h;
            }
            else
            {
                // Best-effort fallback: map tensor-space center/size back to image-space with the same affine.
                float2 centerImage = BlazeUtils.mul(_tensorToImage, new float2(cx, cy));
                float2 topRightImage = BlazeUtils.mul(_tensorToImage, new float2(cx + 0.5f * w, cy + 0.5f * h));
                float2 sizeImage = 2f * (topRightImage - centerImage);

                normCx = centerImage.x / Mathf.Max(1f, _currentFrameWidth);
                normCy = centerImage.y / Mathf.Max(1f, _currentFrameHeight);
                normW = math.abs(sizeImage.x) / Mathf.Max(1f, _currentFrameWidth);
                normH = math.abs(sizeImage.y) / Mathf.Max(1f, _currentFrameHeight);
            }

            if (!TryBuildFace(normCx, normCy, normW, normH, confidence, out var face))
                continue;

            faceList.Add(face);
        }

        if (faceList.Count > 1)
            faceList.Sort((a, b) => b.confidence.CompareTo(a.confidence));

        if (faceList.Count > maxFacesToEmit)
            faceList.RemoveRange(maxFacesToEmit, faceList.Count - maxFacesToEmit);

        return faceList.ToArray();
    }

    private bool TryBuildFace(float normCx, float normCy, float normW, float normH, float confidence, out DetectedFace face)
    {
        face = default;

        if (normW <= minBoxSizeNormalized || normH <= minBoxSizeNormalized)
            return false;

        // Compute bounding box corners first, then clamp bounds-based to avoid overshrinking.
        float left = normCx - normW * 0.5f;
        float top = normCy - normH * 0.5f;
        float right = normCx + normW * 0.5f;
        float bottom = normCy + normH * 0.5f;

        // Clamp bounds to [0, 1] to handle boxes that partially extend outside frame.
        left = Mathf.Clamp01(left);
        top = Mathf.Clamp01(top);
        right = Mathf.Clamp01(right);
        bottom = Mathf.Clamp01(bottom);

        // Convert clamped bounds back to (x, y, width, height) form.
        float x = left;
        float y = top;
        float clampedW = right - left;
        float clampedH = bottom - top;

        if (clampedW <= 0f || clampedH <= 0f)
            return false;

        face = new DetectedFace
        {
            boundingBox = new Rect(x, y, clampedW, clampedH),
            center = new Vector2(normCx, normCy),
            confidence = confidence
        };

        return true;
    }

    private static float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
    }

    private static float[,] TryLoadAnchors(TextAsset csvAsset)
    {
        if (csvAsset == null)
            return null;

        var lines = csvAsset.text.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        var anchors = new float[lines.Length, NumAnchorValues];

        for (int i = 0; i < lines.Length; i++)
        {
            var values = lines[i].Split(',');
            if (values.Length < NumAnchorValues)
            {
                Debug.LogError($"[BlazeFace] Invalid anchors CSV format at line {i + 1}.");
                return null;
            }

            for (int j = 0; j < NumAnchorValues; j++)
            {
                if (!float.TryParse(values[j], NumberStyles.Float, CultureInfo.InvariantCulture, out anchors[i, j]))
                {
                    Debug.LogError($"[BlazeFace] Failed to parse anchor value at line {i + 1}, column {j + 1}.");
                    return null;
                }
            }
        }

        return anchors;
    }

    void OnDestroy()
    {
        inputTensor?.Dispose();
        worker?.Dispose();
    }
}