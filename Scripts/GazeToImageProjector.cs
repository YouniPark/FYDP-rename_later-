using UnityEngine;

public static class GazeToImageProjector
{
    public static bool ProjectRayToPixel(
        Vector3 rayOriginWorld,
        Vector3 rayDirWorld,
        Pose cameraPoseWorld,
        MLCameraFrameProvider.CameraIntrinsics intrinsics,
        int imageWidth,
        int imageHeight,
        out Vector2 pixel)
    {
        pixel = default;
        if (rayDirWorld.sqrMagnitude < 1e-6f) return false;

        var worldToCam = Matrix4x4.TRS(cameraPoseWorld.position, cameraPoseWorld.rotation, Vector3.one).inverse;
        var originCam = worldToCam.MultiplyPoint3x4(rayOriginWorld);
        var dirCam = worldToCam.MultiplyVector(rayDirWorld).normalized;

        const float zPlane = 1.0f;
        if (Mathf.Abs(dirCam.z) < 1e-5f) return false;
        float t = (zPlane - originCam.z) / dirCam.z;
        if (t <= 0f) return false;

        var p = originCam + dirCam * t;

        float fx = intrinsics.Fx;
        float fy = intrinsics.Fy;
        float cx = intrinsics.Cx;
        float cy = intrinsics.Cy;

        if (!intrinsics.IsValid)
        {
            fx = imageWidth * 0.5f;
            fy = imageHeight * 0.5f;
            cx = imageWidth * 0.5f;
            cy = imageHeight * 0.5f;
        }

        float u = fx * (p.x / p.z) + cx;
        float v = fy * (p.y / p.z) + cy;

        pixel = new Vector2(u, imageHeight - v);
        return u >= 0 && u < imageWidth && v >= 0 && v < imageHeight;
    }
}
