using System.Collections.Generic;
using UnityEngine;

public class FaceFixationTracker : MonoBehaviour
{
    public int minDwellMs = 200;
    public int cooldownMs = 300;

    public bool IsFixatingOnFace { get; private set; }
    public int CurrentDwellMs { get; private set; }

    private long _insideStartNs = -1;
    private long _lastTriggerNs = -1;

    public bool UpdateFixation(Vector2 gazePixel, bool gazeValid, long timestampNs, IReadOnlyList<FaceDetectorSentis.Detection> faces, bool fixationEventFlag)
    {
        IsFixatingOnFace = false;
        CurrentDwellMs = 0;

        if (!gazeValid || faces == null || faces.Count == 0)
        {
            _insideStartNs = -1;
            return false;
        }

        bool insideFace = false;
        for (int i = 0; i < faces.Count; i++)
        {
            if (faces[i].PixelRect.Contains(gazePixel))
            {
                insideFace = true;
                break;
            }
        }

        if (!insideFace)
        {
            _insideStartNs = -1;
            return false;
        }

        if (_insideStartNs < 0) _insideStartNs = timestampNs;
        CurrentDwellMs = (int)((timestampNs - _insideStartNs) / 1_000_000L);

        bool cooldownActive = _lastTriggerNs > 0 && ((timestampNs - _lastTriggerNs) / 1_000_000L) < cooldownMs;
        bool dwellTriggered = CurrentDwellMs >= minDwellMs;

        if (!cooldownActive && (dwellTriggered || fixationEventFlag))
        {
            IsFixatingOnFace = true;
            _lastTriggerNs = timestampNs;
            _insideStartNs = timestampNs;
            return true;
        }

        return false;
    }
}
