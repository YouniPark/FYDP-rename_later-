using System.Collections.Generic;
using UnityEngine;

namespace AoiMl2
{
    public sealed class FixationAoiEventEngine
    {
        private readonly float _cooldownSeconds;
        private readonly Dictionary<int, float> _lastHitTime = new Dictionary<int, float>();

        public FixationAoiEventEngine(float cooldownSeconds)
        {
            _cooldownSeconds = cooldownSeconds;
        }

        public bool Step(
            Vector2 gazeNormalized,
            int frameWidth,
            int frameHeight,
            IReadOnlyDictionary<int, RectInt> faceBoxes,
            float timestampSeconds,
            bool isFixation,
            out AoiHitEvent hitEvent)
        {
            hitEvent = default;

            if (!isFixation || frameWidth <= 0 || frameHeight <= 0)
            {
                return false;
            }

            var px = Mathf.Clamp(Mathf.RoundToInt(gazeNormalized.x * (frameWidth - 1)), 0, frameWidth - 1);
            var py = Mathf.Clamp(Mathf.RoundToInt(gazeNormalized.y * (frameHeight - 1)), 0, frameHeight - 1);

            foreach (var pair in faceBoxes)
            {
                var faceId = pair.Key;
                var box = pair.Value;
                if (!box.Contains(new Vector2Int(px, py)))
                {
                    continue;
                }

                _lastHitTime.TryGetValue(faceId, out var lastHit);
                if (timestampSeconds - lastHit < _cooldownSeconds)
                {
                    continue;
                }

                hitEvent = new AoiHitEvent
                {
                    FaceId = faceId,
                    Box = box,
                    GazeNormalized = gazeNormalized,
                    EntryTimestamp = timestampSeconds,
                    EmitTimestamp = timestampSeconds
                };
                _lastHitTime[faceId] = timestampSeconds;
                return true;
            }

            return false;
        }
    }
}
