using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// Streams camera frames from ImageStream to the Python face service over WebSocket.
///
/// This component intentionally stays isolated from cue logic so it can be enabled/disabled
/// independently during integration testing.
/// </summary>
public class FaceVideoStreamSender : MonoBehaviour
{
    [Serializable]
    private class VideoFrameWsMessage
    {
        public double timestamp;
        public string encoding;
        public string data_b64;
    }

    [Header("Enable")]
    [SerializeField, Tooltip("Master switch for sending camera frames to backend.")]
    private bool enableStreaming = true;

    [SerializeField, Tooltip("Current implementation supports WebSocket. WebRTC can be added later.")]
    private FaceVideoTransportMode videoTransport = FaceVideoTransportMode.WebSocket;

    [Header("Shared Backend Config")]
    [SerializeField, Tooltip("Preferred shared config source. If left empty, the first CueConnectionManager in scene is used, then legacy endpoint fields below are used.")]
    private CueConnectionManager sharedConnectionManager;

    [Header("References")]
    [SerializeField, Tooltip("Preferred source of truth for the CV stream. When assigned, frames are sent from the same ImageStream used by this detector.")]
    private BlazeFaceDetector faceDetector;

    [SerializeField, Tooltip("ImageStream that provides latest RGBA frame bytes from ML2 camera.")]
    private ImageStream imageStream;

    [Header("Legacy Fallback Endpoint")]
    [SerializeField, Tooltip("Used only when no BackendConnectionConfig is assigned or found in the scene.")]
    private string serverHost = "10.0.0.1";
    [SerializeField, Tooltip("Used only when no BackendConnectionConfig is assigned or found in the scene.")]
    private int serverPort = 8001;
    [SerializeField, Tooltip("Used only when no BackendConnectionConfig is assigned or found in the scene.")]
    private string wsPath = "/ws/video";

    [Header("Send Settings")]
    [SerializeField, Tooltip("Frame send rate. Keep this low to control bandwidth and backend load.")]
    private float sendFps = 5f;

    [SerializeField, Range(1, 100), Tooltip("JPEG quality used for outgoing frame encoding.")]
    private int jpegQuality = 75;

    [SerializeField, Tooltip("Target encoded frame width. Use <= 0 to keep source width.")]
    private int targetWidth = 640;

    [SerializeField, Tooltip("Target encoded frame height. Use <= 0 to keep source height.")]
    private int targetHeight = 360;

    [Header("Connection")]
    [SerializeField, Tooltip("Seconds before retrying after disconnect/error.")]
    private float reconnectDelaySeconds = 2f;

    [SerializeField, Tooltip("When true, emits periodic debug logs for integration.")]
    private bool verboseLogs = true;

    private ClientWebSocket socket;
    private CancellationTokenSource socketTokenSource;
    private Coroutine reconnectLoopCoroutine;

    private Texture2D sourceTexture;
    private Texture2D resizedTexture;
    private RenderTexture resizeRenderTexture;

    private bool connectInProgress;
    private bool sendInProgress;
    private bool warnedAboutWebRtc;

    private float sendIntervalSeconds;
    private float nextSendTime;
    private long sentFrameCount;

    private void Awake()
    {
        if (sharedConnectionManager == null)
        {
            sharedConnectionManager = FindFirstObjectByType<CueConnectionManager>();
        }

        ResolveImageStream();

        sendIntervalSeconds = 1f / Mathf.Max(0.5f, sendFps);
    }

    private void OnEnable()
    {
        sendIntervalSeconds = 1f / Mathf.Max(0.5f, sendFps);

        if (enableStreaming)
        {
            StartReconnectLoop();
        }
    }

    private void OnDisable()
    {
        StopReconnectLoop();
        _ = CloseSocketAsync("Component disabled");
    }

    private void OnDestroy()
    {
        DestroyEncodeResources();
    }

    private void Update()
    {
        if (!enableStreaming)
        {
            return;
        }

        if (videoTransport != FaceVideoTransportMode.WebSocket)
        {
            if (!warnedAboutWebRtc)
            {
                warnedAboutWebRtc = true;
                Debug.LogWarning("[FaceVideoStreamSender] WebRtc mode selected, but only WebSocket is implemented currently.");
            }
            return;
        }

        if (!IsSocketOpen() || sendInProgress)
        {
            return;
        }

        if (Time.unscaledTime < nextSendTime)
        {
            return;
        }

        ResolveImageStream();

        if (imageStream == null)
        {
            if (verboseLogs)
            {
                Debug.LogWarning("[FaceVideoStreamSender] ImageStream reference missing; cannot send frames.");
            }
            return;
        }

        if (!imageStream.TryGetLatestFrameRgba(out byte[] rgbaFrame, out int width, out int height))
        {
            return;
        }

        nextSendTime = Time.unscaledTime + sendIntervalSeconds;
        _ = SendFrameAsync(rgbaFrame, width, height);
    }

    private void StartReconnectLoop()
    {
        if (reconnectLoopCoroutine == null)
        {
            reconnectLoopCoroutine = StartCoroutine(ReconnectLoop());
        }
    }

    private void StopReconnectLoop()
    {
        if (reconnectLoopCoroutine != null)
        {
            StopCoroutine(reconnectLoopCoroutine);
            reconnectLoopCoroutine = null;
        }
    }

    private System.Collections.IEnumerator ReconnectLoop()
    {
        while (enabled && gameObject.activeInHierarchy)
        {
            if (videoTransport == FaceVideoTransportMode.WebSocket && !IsSocketOpen() && !connectInProgress)
            {
                _ = ConnectSocketAsync();
            }

            yield return new WaitForSeconds(reconnectDelaySeconds);
        }
    }

    private async Task ConnectSocketAsync()
    {
        if (connectInProgress)
        {
            return;
        }

        connectInProgress = true;

        try
        {
            await CloseSocketAsync("Reconnect before open");

            socketTokenSource = new CancellationTokenSource();
            socket = new ClientWebSocket();

            Uri wsUri = BuildWebSocketUri();
            if (verboseLogs)
            {
                Debug.Log($"[FaceVideoStreamSender] Connecting to {wsUri}");
            }

            await socket.ConnectAsync(wsUri, socketTokenSource.Token);

            if (verboseLogs)
            {
                Debug.Log("[FaceVideoStreamSender] WebSocket connected.");
            }

            _ = ReceiveLoopAsync();
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[FaceVideoStreamSender] WebSocket connect failed: {ex.Message}");
        }
        finally
        {
            connectInProgress = false;
        }
    }

    private async Task ReceiveLoopAsync()
    {
        byte[] receiveBuffer = new byte[64];

        try
        {
            while (IsSocketOpen() && socketTokenSource != null && !socketTokenSource.IsCancellationRequested)
            {
                WebSocketReceiveResult result = await socket.ReceiveAsync(
                    new ArraySegment<byte>(receiveBuffer),
                    socketTokenSource.Token);

                if (result.MessageType == WebSocketMessageType.Close)
                {
                    break;
                }
            }
        }
        catch (Exception)
        {
            // Receive loop failures are expected during disconnects/reconnects.
        }
        finally
        {
            await CloseSocketAsync("Receive loop ended");
        }
    }

    private async Task SendFrameAsync(byte[] rgbaFrame, int width, int height)
    {
        if (!IsSocketOpen() || socketTokenSource == null)
        {
            return;
        }

        sendInProgress = true;

        try
        {
            byte[] jpegBytes = EncodeFrameToJpeg(rgbaFrame, width, height);
            if (jpegBytes == null || jpegBytes.Length == 0)
            {
                return;
            }

            VideoFrameWsMessage message = new VideoFrameWsMessage
            {
                timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0,
                encoding = "jpeg",
                data_b64 = Convert.ToBase64String(jpegBytes)
            };

            string payloadJson = JsonUtility.ToJson(message);
            byte[] payloadBytes = Encoding.UTF8.GetBytes(payloadJson);

            await socket.SendAsync(
                new ArraySegment<byte>(payloadBytes),
                WebSocketMessageType.Text,
                true,
                socketTokenSource.Token);

            sentFrameCount++;
            if (verboseLogs && sentFrameCount % 30 == 0)
            {
                Debug.Log($"[FaceVideoStreamSender] Sent frames: {sentFrameCount}");
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[FaceVideoStreamSender] Send failed: {ex.Message}");
            await CloseSocketAsync("Send failed");
        }
        finally
        {
            sendInProgress = false;
        }
    }

    private byte[] EncodeFrameToJpeg(byte[] rgbaFrame, int width, int height)
    {
        EnsureSourceTexture(width, height);

        sourceTexture.LoadRawTextureData(rgbaFrame);
        sourceTexture.Apply(false, false);

        Texture2D encodeTexture = sourceTexture;

        // Downsample before JPEG encoding if a target size is configured.
        if (targetWidth > 0 && targetHeight > 0 && (targetWidth != width || targetHeight != height))
        {
            EnsureResizeResources(targetWidth, targetHeight);

            Graphics.Blit(sourceTexture, resizeRenderTexture);

            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = resizeRenderTexture;

            resizedTexture.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0, false);
            resizedTexture.Apply(false, false);

            RenderTexture.active = previousActive;
            encodeTexture = resizedTexture;
        }

        return encodeTexture.EncodeToJPG(jpegQuality);
    }

    private void EnsureSourceTexture(int width, int height)
    {
        if (sourceTexture != null && sourceTexture.width == width && sourceTexture.height == height)
        {
            return;
        }

        if (sourceTexture != null)
        {
            Destroy(sourceTexture);
        }

        sourceTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
    }

    private void EnsureResizeResources(int width, int height)
    {
        bool needsTextureRecreate = resizedTexture == null || resizedTexture.width != width || resizedTexture.height != height;
        bool needsRenderTextureRecreate = resizeRenderTexture == null || resizeRenderTexture.width != width || resizeRenderTexture.height != height;

        if (needsTextureRecreate)
        {
            if (resizedTexture != null)
            {
                Destroy(resizedTexture);
            }
            resizedTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
        }

        if (needsRenderTextureRecreate)
        {
            if (resizeRenderTexture != null)
            {
                resizeRenderTexture.Release();
                Destroy(resizeRenderTexture);
            }

            resizeRenderTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            resizeRenderTexture.Create();
        }
    }

    private async Task CloseSocketAsync(string reason)
    {
        try
        {
            if (socket != null)
            {
                if (socket.State == WebSocketState.Open)
                {
                    await socket.CloseAsync(
                        WebSocketCloseStatus.NormalClosure,
                        reason,
                        CancellationToken.None);
                }

                socket.Dispose();
            }
        }
        catch (Exception)
        {
            // Ignore close errors during teardown.
        }
        finally
        {
            socket = null;

            if (socketTokenSource != null)
            {
                socketTokenSource.Cancel();
                socketTokenSource.Dispose();
                socketTokenSource = null;
            }
        }
    }

    private Uri BuildWebSocketUri()
    {
        if (sharedConnectionManager != null)
        {
            return new Uri(sharedConnectionManager.BuildVideoWebSocketUrl());
        }

        string normalizedPath = string.IsNullOrWhiteSpace(wsPath) ? "/ws/video" : wsPath.Trim();
        if (!normalizedPath.StartsWith("/"))
        {
            normalizedPath = "/" + normalizedPath;
        }

        string host = string.IsNullOrWhiteSpace(serverHost) ? "127.0.0.1" : serverHost.Trim();

        if (host.StartsWith("ws://", StringComparison.OrdinalIgnoreCase) ||
            host.StartsWith("wss://", StringComparison.OrdinalIgnoreCase))
        {
            return new Uri(host.TrimEnd('/') + normalizedPath);
        }

        return new Uri($"ws://{host}:{serverPort}{normalizedPath}");
    }

    private bool IsSocketOpen()
    {
        return socket != null && socket.State == WebSocketState.Open;
    }

    private void ResolveImageStream()
    {
        if (faceDetector == null)
        {
            faceDetector = FindFirstObjectByType<BlazeFaceDetector>();
        }

        ImageStream detectorStream = faceDetector != null ? faceDetector.CameraSource : null;
        if (detectorStream != null)
        {
            imageStream = detectorStream;
            return;
        }

        if (imageStream == null)
        {
            imageStream = FindFirstObjectByType<ImageStream>();
        }
    }

    private void DestroyEncodeResources()
    {
        if (sourceTexture != null)
        {
            Destroy(sourceTexture);
            sourceTexture = null;
        }

        if (resizedTexture != null)
        {
            Destroy(resizedTexture);
            resizedTexture = null;
        }

        if (resizeRenderTexture != null)
        {
            resizeRenderTexture.Release();
            Destroy(resizeRenderTexture);
            resizeRenderTexture = null;
        }
    }
}
