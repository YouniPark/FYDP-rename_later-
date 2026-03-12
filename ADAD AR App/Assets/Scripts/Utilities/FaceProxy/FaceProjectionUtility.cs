using UnityEngine;
using UnityEngine.XR.MagicLeap;

/// <summary>
/// Utility functions for projecting normalized camera-frame points into world space.
/// </summary>
public static class FaceProjectionUtility
{
    /// <summary>
    /// Projects a normalized viewport point (0..1) to a world-space point at a fixed depth along the camera ray.
    /// </summary>
    public static Vector3 ProjectViewportPointAtDepth(
        MLCamera.IntrinsicCalibrationParameters intrinsics,
        Matrix4x4 framePose,
        Vector2 viewportPoint,
        float depth,
        Transform trackingSpaceTransform = null,
        Transform headTransform = null)
    {
        Matrix4x4 worldPose = ResolveWorldPose(framePose, trackingSpaceTransform, headTransform);
        Vector2 undistorted = UndistortViewportPoint(intrinsics, viewportPoint);
        Ray ray = RayFromViewportPoint(intrinsics, undistorted, worldPose.GetPosition(), worldPose.rotation);
        return ray.GetPoint(Mathf.Max(0.01f, depth));
    }

    /// <summary>
    /// Resolves the world-space camera pose for projection.
    /// Prefers the live head transform when available, otherwise falls back to tracking-space conversion.
    /// </summary>
    public static Matrix4x4 ResolveWorldPose(Matrix4x4 framePose, Transform trackingSpaceTransform, Transform headTransform)
    {
        if (headTransform != null)
        {
            return Matrix4x4.TRS(headTransform.position, headTransform.rotation, Vector3.one);
        }

        return ConvertTrackingPoseToWorld(framePose, trackingSpaceTransform);
    }

    /// <summary>
    /// Converts a camera pose from tracking-space to world-space.
    /// If trackingSpaceTransform is null, framePose is assumed to already be world-space.
    /// </summary>
    public static Matrix4x4 ConvertTrackingPoseToWorld(Matrix4x4 framePose, Transform trackingSpaceTransform)
    {
        if (trackingSpaceTransform == null)
        {
            return framePose;
        }

        return trackingSpaceTransform.localToWorldMatrix * framePose;
    }

    /// <summary>
    /// Undistorts a viewport point to account for camera lens distortion.
    /// </summary>
    public static Vector2 UndistortViewportPoint(MLCamera.IntrinsicCalibrationParameters intrinsics, Vector2 distortedViewportPoint)
    {
        float width = Mathf.Max(1f, intrinsics.Width);
        float height = Mathf.Max(1f, intrinsics.Height);

        float normalizedToPixel = new Vector2(width / 2f, height / 2f).magnitude;
        float pixelToNormalized = Mathf.Approximately(normalizedToPixel, 0f) ? float.MaxValue : 1f / normalizedToPixel;
        Vector2 viewportToNormalized = new Vector2(width * pixelToNormalized, height * pixelToNormalized);
        Vector2 normalizedPrincipalPoint = intrinsics.PrincipalPoint * pixelToNormalized;
        Vector2 normalizedToViewport = new Vector2(
            Mathf.Approximately(viewportToNormalized.x, 0f) ? 0f : 1f / viewportToNormalized.x,
            Mathf.Approximately(viewportToNormalized.y, 0f) ? 0f : 1f / viewportToNormalized.y);

        Vector2 d = Vector2.Scale(distortedViewportPoint, viewportToNormalized);
        Vector2 o = d - normalizedPrincipalPoint;

        float k1 = intrinsics.Distortion.Length > 0 ? (float)intrinsics.Distortion[0] : 0f;
        float k2 = intrinsics.Distortion.Length > 1 ? (float)intrinsics.Distortion[1] : 0f;
        float p1 = intrinsics.Distortion.Length > 2 ? (float)intrinsics.Distortion[2] : 0f;
        float p2 = intrinsics.Distortion.Length > 3 ? (float)intrinsics.Distortion[3] : 0f;
        float k3 = intrinsics.Distortion.Length > 4 ? (float)intrinsics.Distortion[4] : 0f;

        float r2 = o.sqrMagnitude;
        float r4 = r2 * r2;
        float r6 = r2 * r4;
        float radial = k1 * r2 + k2 * r4 + k3 * r6;

        Vector2 undistorted = d + o * radial;

        if (!Mathf.Approximately(p1, 0f) || !Mathf.Approximately(p2, 0f))
        {
            undistorted.x += p1 * (r2 + 2f * o.x * o.x) + 2f * p2 * o.x * o.y;
            undistorted.y += p2 * (r2 + 2f * o.y * o.y) + 2f * p1 * o.x * o.y;
        }

        return Vector2.Scale(undistorted, normalizedToViewport);
    }

    /// <summary>
    /// Creates a world-space ray from a viewport point using physical camera intrinsics.
    /// </summary>
    public static Ray RayFromViewportPoint(
        MLCamera.IntrinsicCalibrationParameters intrinsics,
        Vector2 viewportPoint,
        Vector3 cameraPosition,
        Quaternion cameraRotation)
    {
        float width = Mathf.Max(1f, intrinsics.Width);
        float height = Mathf.Max(1f, intrinsics.Height);

        Vector2 pixel = new Vector2(viewportPoint.x * width, viewportPoint.y * height);
        Vector2 offset = new Vector2(
            pixel.x - intrinsics.PrincipalPoint.x,
            pixel.y - (height - intrinsics.PrincipalPoint.y));

        float fx = Mathf.Approximately(intrinsics.FocalLength.x, 0f) ? 1f : intrinsics.FocalLength.x;
        float fy = Mathf.Approximately(intrinsics.FocalLength.y, 0f) ? 1f : intrinsics.FocalLength.y;

        Vector2 unitFocal = new Vector2(offset.x / fx, offset.y / fy);
        Vector3 direction = cameraRotation * new Vector3(unitFocal.x, unitFocal.y, 1f).normalized;

        return new Ray(cameraPosition, direction);
    }
}
