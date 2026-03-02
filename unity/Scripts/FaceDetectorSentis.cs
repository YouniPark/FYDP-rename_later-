using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Sentis;

public class FaceDetectorSentis : MonoBehaviour
{
    [Serializable]
    public struct Detection
    {
        public Rect PixelRect;
        public float Score;
    }

    [Header("BlazeFace (Sentis)")]
    public ModelAsset blazeFaceModel;
    public int inputWidth = 128;
    public int inputHeight = 128;
    [Range(0f, 1f)] public float scoreThreshold = 0.6f;

    [Header("Timing")]
    [Tooltip("Max detections per second. 10 is a good starting point on ML2.")]
    public float detectionHz = 10f;

    [Tooltip("If true, runs detection automatically at detectionHz when a texture is provided via SetSourceTexture().")]
    public bool autoRun = true;

    private Model _runtimeModel;
    private Worker _worker;
    private Tensor<float> _inputTensor;

    private bool _isBusy;
    private float _nextDetectTime;

    private Texture _sourceTexture;

    private readonly List<Detection> _latest = new List<Detection>(32);
    public IReadOnlyList<Detection> LatestDetections => _latest;
    public bool HasDetections => _latest.Count > 0;

    public bool IsReady => _worker != null;
    public bool IsBusy => _isBusy;

    private void Awake()
    {
        if (blazeFaceModel == null)
        {
            Debug.LogError("FaceDetectorSentis: missing BlazeFace ModelAsset.");
            return;
        }

        _runtimeModel = ModelLoader.Load(blazeFaceModel);
        Debug.Log("FaceDetectorSentis: Model outputs:");
        foreach (var o in _runtimeModel.outputs)
            Debug.Log($"FaceDetectorSentis: OUT = {o.name}");


        _worker = new Worker(_runtimeModel, BackendType.GPUCompute);

        _inputTensor = new Tensor<float>(new TensorShape(1, inputHeight, inputWidth, 3));
        _nextDetectTime = 0f;

        Debug.Log("FaceDetectorSentis: Awake OK (model loaded, worker created).");
    }

    private void OnDestroy()
    {
        _inputTensor?.Dispose();
        _worker?.Dispose();
    }

    /// <summary>
    /// Provide a texture to run detections on. OverlayRenderer can call this every frame.
    /// </summary>
    public void SetSourceTexture(Texture tex)
    {
        _sourceTexture = tex;
    }

    private async void Update()
    {
        if (!autoRun) return;
        if (_sourceTexture == null) return;
        if (_worker == null) return;
        if (_isBusy) return;

        // Throttle
        float hz = Mathf.Max(1f, detectionHz);
        if (Time.time < _nextDetectTime) return;
        _nextDetectTime = Time.time + (1f / hz);

        // Run and cache
        await DetectAndCacheAsync(_sourceTexture);
    }

    /// <summary>
    /// Runs detection and updates LatestDetections cache.
    /// </summary>
    public async Task<List<Detection>> DetectAndCacheAsync(Texture sourceTexture)
    {
        var results = await DetectAsync(sourceTexture);

        _latest.Clear();
        _latest.AddRange(results);

        return results;
    }

    /// <summary>
    /// One-shot detection. Returns detections in pixel coordinates of the source texture.
    /// NOTE: Uses synchronous ReadbackAndClone() (compatible with older Sentis).
    /// </summary>
    public async Task<List<Detection>> DetectAsync(Texture sourceTexture)
    {
        await Task.Yield();

        var results = new List<Detection>(16);
        if (_worker == null || sourceTexture == null || _isBusy)
            return results;

        _isBusy = true;

        try
        {
            // Must run on main thread
            TextureConverter.ToTensor(sourceTexture, _inputTensor, new TextureTransform());
            _worker.Schedule(_inputTensor);

            Tensor<float> boxes = null;
            Tensor<float> scores = null;

            foreach (var output in _runtimeModel.outputs)
            {
                var n = output.name.ToLowerInvariant();
                if (boxes == null && n.Contains("regressors")) boxes = _worker.PeekOutput(output.name) as Tensor<float>;
                if (scores == null && n.Contains("classifiers")) scores = _worker.PeekOutput(output.name) as Tensor<float>;
            }

            if (boxes == null || scores == null)
            {
                Debug.LogError("FaceDetectorSentis: could not find expected boxes/scores outputs. Check your model output names.");
                return results;
            }

            // Older Sentis: sync readback
            using (var readableBoxes = boxes.ReadbackAndClone())
            using (var readableScores = scores.ReadbackAndClone())
            {
                ParseOutputs(readableBoxes, readableScores, sourceTexture.width, sourceTexture.height, results);
            }
        }
        finally
        {
            _isBusy = false;
        }

        return results;
    }

    private void ParseOutputs(Tensor<float> boxes, Tensor<float> scores, int imageWidth, int imageHeight, List<Detection> outDetections)
    {
        int count = Mathf.Min(scores.shape.length, boxes.shape.length / 4);

        for (int i = 0; i < count; i++)
        {
            float score = scores[i];
            if (score < scoreThreshold) continue;

            int b = i * 4;
            float yMin = Mathf.Clamp01(boxes[b + 0]);
            float xMin = Mathf.Clamp01(boxes[b + 1]);
            float yMax = Mathf.Clamp01(boxes[b + 2]);
            float xMax = Mathf.Clamp01(boxes[b + 3]);

            float x = xMin * imageWidth;
            float y = (1f - yMax) * imageHeight; // top-left origin UI coords
            float w = Mathf.Max(1f, (xMax - xMin) * imageWidth);
            float h = Mathf.Max(1f, (yMax - yMin) * imageHeight);

            outDetections.Add(new Detection { PixelRect = new Rect(x, y, w, h), Score = score });
        }
    }
}
