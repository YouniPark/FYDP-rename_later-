using System;
using UnityEngine;

[DisallowMultipleComponent]
public class BackendConnectionConfig : MonoBehaviour
{
    [Header("Backend Address")]
    [SerializeField, Tooltip("IPv4 or host name for the Python backend. Change this once here and all linked components will use it.")]
    private string serverHost = "192.168.4.29";

    [SerializeField, Tooltip("Backend port shared by WebSocket video ingest and HTTP face lookup.")]
    private int serverPort = 8001;

    [SerializeField, Tooltip("Enable only if the backend is explicitly exposed over TLS.")]
    private bool useTls = false;

    [Header("Paths")]
    [SerializeField] private string videoWsPath = "/ws/video";
    [SerializeField] private string faceLookupPath = "/face/latest";

    public string BuildFaceLookupUrl()
    {
        string scheme = useTls ? "https" : "http";
        return $"{scheme}://{GetHost()}:{Mathf.Max(1, serverPort)}{NormalizePath(faceLookupPath, "/face/latest")}";
    }

    public Uri BuildVideoWebSocketUri()
    {
        string scheme = useTls ? "wss" : "ws";
        return new Uri($"{scheme}://{GetHost()}:{Mathf.Max(1, serverPort)}{NormalizePath(videoWsPath, "/ws/video")}");
    }

    private string GetHost()
    {
        return string.IsNullOrWhiteSpace(serverHost) ? "127.0.0.1" : serverHost.Trim();
    }

    private static string NormalizePath(string rawPath, string defaultPath)
    {
        string normalizedPath = string.IsNullOrWhiteSpace(rawPath) ? defaultPath : rawPath.Trim();
        if (!normalizedPath.StartsWith("/"))
        {
            normalizedPath = "/" + normalizedPath;
        }

        return normalizedPath;
    }
}