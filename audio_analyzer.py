import wave
import audioop
import numpy as np

BAD_WORDS = [
    "fuck", "shit", "bitch", "bastard", "asshole",
    "whore", "nigger", "faggot", "kill", "die"
]

def detect_bad_words(transcript_text: str):
    text = transcript_text.lower()
    flags = [w for w in BAD_WORDS if w in text]
    return flags


# -------- SIMPLE PITCH ESTIMATION (child voice detection) -------- #

def estimate_pitch(frame_bytes, sample_rate):
    try:
        # Zero crossing rate → approximates frequency
        crossings = np.where(np.diff(np.sign(frame_bytes)))[0]
        if len(crossings) < 2:
            return 0
        periods = np.diff(crossings)
        if len(periods) == 0:
            return 0
        avg_period = np.mean(periods)
        freq = sample_rate / avg_period
        return freq
    except:
        return 0


def detect_child_voice(pitches):
    # Children typically have higher pitch > 260 Hz
    if np.mean(pitches) > 260:
        return True
    return False


# -------- ENERGY / SHOUT DETECTION -------- #

def detect_aggression(energies):
    # If audio has sudden loudness spikes → shouting/aggression
    if len(energies) == 0:
        return False

    mean_energy = np.mean(energies)
    max_energy = np.max(energies)

    if max_energy > mean_energy * 5:
        return True

    return False


# -------- MAIN AUDIO ANALYZER -------- #

def analyze_audio(audio_path: str):
    try:
        audio = wave.open(audio_path, "rb")
    except:
        return {"status": "error", "detail": "Audio could not be read"}

    sample_rate = audio.getframerate()
    frames = audio.getnframes()
    duration = frames / sample_rate

    energies = []
    pitches = []

    chunk_size = 2048

    while True:
        frame = audio.readframes(chunk_size)
        if not frame:
            break

        # Convert to mono for analysis
        try:
            mono = audioop.tomono(frame, 2, 1, 0)
        except:
            continue

        # Energy estimation
        try:
            energy = audioop.rms(mono, 2)
            energies.append(energy)
        except:
            pass

        # Convert to int16 array for pitch
        try:
            data = np.frombuffer(mono, dtype=np.int16)
            pitch = estimate_pitch(data, sample_rate)
            if pitch > 50:  # ignore meaningless values
                pitches.append(pitch)
        except:
            pass

    audio.close()

    return {
        "audio_flags": {
            "child_voice": detect_child_voice(pitches),
            "aggression": detect_aggression(energies),
        }
    }
