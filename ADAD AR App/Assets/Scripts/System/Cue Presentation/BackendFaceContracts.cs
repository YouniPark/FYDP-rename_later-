using System;
using UnityEngine;

[Serializable]
public class BackendLatestFaceResponse
{
    public string name;

    // Known person ID, 0 for Unknown, null when there is no face decision.
    public int? people_id;

    // Null when there is no face decision available.
    public float? confidence;

    // UTC ISO-8601 timestamp from server.
    public string decided_at;

    public string source;
    public float window_seconds;
    public int sample_count;
    public bool is_unknown;
}

public enum FaceVideoTransportMode
{
    WebSocket = 0,
    WebRtc = 1,
}

[Serializable]
public class BackendFaceLookupConfig
{
    public FaceVideoTransportMode videoTransport = FaceVideoTransportMode.WebSocket;
    public float inferenceSampleFps = 5f;
    public float memoryWindowSeconds = 2f;
    public float unknownThreshold = 0.5f;
    public int jpegQuality = 75;
    [HideInInspector]
    public string videoWsUrl = "ws://192.168.4.29:8001/ws/video";
    [HideInInspector]
    public string faceLookupUrl = "http://192.168.4.29:8001/face/latest";
    public float requestTimeoutSeconds = 3f;
    public string tieBreakStrategy = "most_recent";
}
