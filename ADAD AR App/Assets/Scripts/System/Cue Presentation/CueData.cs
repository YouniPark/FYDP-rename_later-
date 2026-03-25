using System;

/// <summary>
/// Serializable data model matching the cue JSON payload (from backend or local Resources/).
/// Example JSON structure:
/// {
///   "people_id": 1,
///   "font_size_px": 48,
///   "image_scale": 1.0,
///   "duration_seconds": 90,
///   "cues": {
///     "name": "Person",
///     "relationship": "Friend",
///     "image_path": "Cue Defaults/1/cue_photo_daniel_1",  // optional, Resources-relative, no extension
///     "audio_path": "Cue Defaults/1/cue_audio_daniel_male", // optional, Resources-relative, no extension
///     "image_url": "http://10.0.0.1:5000/image/1",         // optional, fallback network download
///     "audio_url": "http://10.0.0.1:5000/audio/1"          // optional, fallback network download
///   }
/// }
/// Resolution priority for images: inspector override > image_path (Resources) > image_url (network)
/// Resolution priority for audio:  inspector override > audio_path (Resources) > audio_url (network)
/// </summary>
[Serializable]
public class CueData
{
    public int   people_id        = -1;
    public int   font_size_px     = 48;
    public float image_scale      = 1.0f;
    public float duration_seconds = 60f;

    public CueDetails cues;

    [Serializable]
    public class CueDetails
    {
        public string name         = "Person";
        public string relationship = "Relationship";
        /// <summary>Resources-relative path (no extension) for local Texture2D/Sprite. Checked before image_url.</summary>
        public string image_path;
        /// <summary>Resources-relative path (no extension) for local AudioClip. Checked before audio_url.</summary>
        public string audio_path;
        /// <summary>Optional URL served by backend. Used only when image_path is absent.</summary>
        public string image_url;
        /// <summary>Optional URL served by backend (mp3 / ogg / wav). Used only when audio_path is absent.</summary>
        public string audio_url;
    }
}
