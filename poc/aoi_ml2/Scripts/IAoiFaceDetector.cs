using System.Collections.Generic;
using UnityEngine;

namespace AoiMl2
{
    public interface IAoiFaceDetector
    {
        IReadOnlyDictionary<int, RectInt> DetectFaces(Texture2D cameraFrame);
    }
}
