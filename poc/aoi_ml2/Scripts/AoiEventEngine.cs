using System;
using System.Collections.Generic;
using UnityEngine;

namespace AoiMl2
{
    public sealed class AoiEventEngine
    {
        private readonly float _minDwellSeconds;
        private readonly float _cooldownSeconds;
        private readonly Dictionary<int, bool> _entered = new Dictionary<int, bool>();
        private readonly Dictionary<int, float?> _entryTime = new Dictionary<int, float?>();
        private readonly Dictionary<int, float> _lastHitTime = new Dictionary<int, float>();

        public AoiEventEngine(float minDwellSeconds, float cooldownSeconds)
        {
            _minDwellSeconds = minDwellSeconds;
            _cooldownSeconds = cooldownSeconds;
        }

        public bool Step(
            Vector2 gazeNormalized,
            int frameWidth,
            int frameHeight,
            IReadOnlyDictionary<int, RectInt> faceBoxes,
            float timestampSeconds,
            out AoiHitEvent hitEvent)
        {
            hitEvent = default;

            if (frameWidth <= 0 || frameHeight <= 0)
            {
                return false;
            }

            var px = Mathf.Clamp(Mathf.RoundToInt(gazeNormalized.x * (frameWidth - 1)), 0, frameWidth - 1);
            var py = Mathf.Clamp(Mathf.RoundToInt(gazeNormalized.y * (frameHeight - 1)), 0, frameHeight - 1);

            foreach (var pair in faceBoxes)
            {
                var faceId = pair.Key;
                var box = pair.Value;
                var isInside = box.Contains(new Vector2Int(px, py));
                var wasInside = _entered.TryGetValue(faceId, out var prevInside) && prevInside;

                if (isInside && !wasInside)
                {
                    _entered[faceId] = true;
                    _entryTime[faceId] = timestampSeconds;
                }
                else if (isInside && wasInside)
                {
                    if (_entryTime.TryGetValue(faceId, out var entryTimestamp) && entryTimestamp.HasValue)
                    {
                        var dwellDuration = timestampSeconds - entryTimestamp.Value;
                        if (dwellDuration >= _minDwellSeconds)
                        {
                            _lastHitTime.TryGetValue(faceId, out var lastHit);
                            if (timestampSeconds - lastHit >= _cooldownSeconds)
                            {
                                hitEvent = new AoiHitEvent
                                {
                                    FaceId = faceId,
                                    Box = box,
                                    GazeNormalized = gazeNormalized,
                                    EntryTimestamp = entryTimestamp.Value,
                                    EmitTimestamp = timestampSeconds
                                };
                                _lastHitTime[faceId] = timestampSeconds;
                                _entryTime[faceId] = null;
                                return true;
                            }
                        }
                    }
                }
                else if (!isInside && wasInside)
                {
                    _entered[faceId] = false;
                    _entryTime[faceId] = null;
                }
            }

            return false;
        }
    }

    public readonly struct AoiHitEvent
    {
        public int FaceId { get; init; }
        public RectInt Box { get; init; }
        public Vector2 GazeNormalized { get; init; }
        public float EntryTimestamp { get; init; }
        public float EmitTimestamp { get; init; }
    }
}
