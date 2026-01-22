using System.Collections.Generic;
using System.IO;
using UnityEngine;

#if OPENCV_FOR_UNITY
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
#endif

namespace AoiMl2
{
    public sealed class OpenCvDnnFaceDetector : MonoBehaviour, IAoiFaceDetector
    {
        [Header("Model Settings")]
        [SerializeField] private string prototxtFileName = "deploy.prototxt";
        [SerializeField] private string weightsFileName = "res10_300x300_ssd_iter_140000.caffemodel";
        [SerializeField, Range(0.1f, 0.99f)] private float confidenceThreshold = 0.6f;

        private readonly Dictionary<int, RectInt> _faces = new Dictionary<int, RectInt>();

#if OPENCV_FOR_UNITY
        private Net _net;
        private Mat _rgbaMat;
        private Mat _bgrMat;
        private Mat _inputBlob;
        private readonly Size _inputSize = new Size(300, 300);
        private readonly Scalar _mean = new Scalar(104.0, 177.0, 123.0);
#else
        private bool _loggedMissing;
#endif

        private void OnEnable()
        {
#if OPENCV_FOR_UNITY
            var prototxtPath = ResolveModelPath(prototxtFileName);
            var weightsPath = ResolveModelPath(weightsFileName);
            _net = Dnn.readNetFromCaffe(prototxtPath, weightsPath);
#endif
        }

        private void OnDisable()
        {
#if OPENCV_FOR_UNITY
            _net?.Dispose();
            _net = null;
            _rgbaMat?.Dispose();
            _bgrMat?.Dispose();
            _inputBlob?.Dispose();
#endif
        }

        public IReadOnlyDictionary<int, RectInt> DetectFaces(Texture2D cameraFrame)
        {
            _faces.Clear();

#if OPENCV_FOR_UNITY
            if (cameraFrame == null || _net == null)
            {
                return _faces;
            }

            EnsureMats(cameraFrame.width, cameraFrame.height);
            Utils.texture2DToMat(cameraFrame, _rgbaMat);
            Imgproc.cvtColor(_rgbaMat, _bgrMat, Imgproc.COLOR_RGBA2BGR);

            _inputBlob?.Dispose();
            _inputBlob = Dnn.blobFromImage(_bgrMat, 1.0, _inputSize, _mean, false, false);
            _net.setInput(_inputBlob);
            using var detections = _net.forward();

            var width = _bgrMat.width();
            var height = _bgrMat.height();
            var detectionCount = detections.size(2);
            var detectionData = new float[detections.total() * detections.channels()];
            detections.get(0, 0, detectionData);

            var index = 0;
            for (var i = 0; i < detectionCount; i++)
            {
                var confidence = detectionData[index + 2];
                if (confidence >= confidenceThreshold)
                {
                    var x1 = Mathf.Clamp(Mathf.RoundToInt(detectionData[index + 3] * width), 0, width - 1);
                    var y1 = Mathf.Clamp(Mathf.RoundToInt(detectionData[index + 4] * height), 0, height - 1);
                    var x2 = Mathf.Clamp(Mathf.RoundToInt(detectionData[index + 5] * width), 0, width - 1);
                    var y2 = Mathf.Clamp(Mathf.RoundToInt(detectionData[index + 6] * height), 0, height - 1);
                    var rect = new RectInt(x1, y1, Mathf.Max(1, x2 - x1), Mathf.Max(1, y2 - y1));
                    _faces[i] = rect;
                }

                index += 7;
            }

            return _faces;
#else
            if (!_loggedMissing)
            {
                Debug.LogWarning("OpenCvDnnFaceDetector requires the OpenCV for Unity plugin. Returning no detections.");
                _loggedMissing = true;
            }

            return _faces;
#endif
        }

#if OPENCV_FOR_UNITY
        private void EnsureMats(int width, int height)
        {
            if (_rgbaMat != null && _rgbaMat.width() == width && _rgbaMat.height() == height)
            {
                return;
            }

            _rgbaMat?.Dispose();
            _bgrMat?.Dispose();
            _rgbaMat = new Mat(height, width, CvType.CV_8UC4);
            _bgrMat = new Mat(height, width, CvType.CV_8UC3);
        }

        private string ResolveModelPath(string fileName)
        {
            if (Path.IsPathRooted(fileName))
            {
                return fileName;
            }

            return Path.Combine(Application.streamingAssetsPath, fileName);
        }
#endif
    }
}
