from collections import deque, Counter

EMOTION_WEIGHTS = {
    'happy':    +1.0,
    'neutral':  +0.5,
    'surprise': +0.3,
    'fear':     -0.3,
    'sad':      -0.5,
    'angry':    -0.7,
    'disgust':  -0.8,
}

LOW_ENGAGEMENT_EMOTIONS = {'sad', 'angry', 'disgust', 'fear'}
ALERT_THRESHOLD = 10

REALTIME_SUGGESTIONS = {
    'angry':   "Take a 5-minute break and come back fresh.",
    'fear':    "You seem stuck. Try revisiting the previous section.",
    'sad':     "Feeling low? A short walk or some water might help.",
    'disgust': "This topic seems frustrating. Try a different resource or example.",
}

SESSION_SUGGESTIONS = {
    'angry':   "You experienced frequent frustration. Consider breaking the material into smaller chunks next time.",
    'fear':    "Confusion was the dominant state. Review the prerequisite concepts before the next session.",
    'sad':     "Low engagement was detected often. Try studying at a different time of day when you feel more alert.",
    'disgust': "Strong disengagement was detected. Try switching to video lectures or a different format for this topic.",
    'neutral': "Good steady focus. Keep your current study routine.",
    'happy':   "Excellent session. You stayed engaged and positive throughout.",
    'surprise':"High alertness detected. You were actively engaged with new material.",
}

class FocusEngine:
    def __init__(self, window_size=30):
        self.window      = deque(maxlen=window_size)
        self.emotion_log = []
        self.alert_log   = []
        self.low_streak  = 0
        self.focus_score = 50.0

    def update(self, emotion, timestamp):
        weight = EMOTION_WEIGHTS.get(emotion, 0)
        self.window.append(weight)
        self.emotion_log.append((timestamp, emotion))

        avg = sum(self.window) / len(self.window)
        self.focus_score = min(100, max(0, 50 + avg * 40))

        if emotion in LOW_ENGAGEMENT_EMOTIONS:
            self.low_streak += 1
        else:
            self.low_streak = 0

        alert = None
        if self.low_streak >= ALERT_THRESHOLD:
            alert = REALTIME_SUGGESTIONS.get(emotion)
            if alert:
                self.alert_log.append((timestamp, alert))
            self.low_streak = 0

        return self.focus_score, alert

    def get_dominant_emotion(self):
        if not self.emotion_log:
            return "neutral"
        counts = Counter(e for _, e in self.emotion_log)
        return counts.most_common(1)[0][0]

    def get_summary(self, total_seconds):
        emotion_counts = Counter(e for _, e in self.emotion_log)
        total          = max(len(self.emotion_log), 1)
        emotion_times  = {e: round(c / total * total_seconds, 1) for e, c in emotion_counts.items()}
        dominant       = self.get_dominant_emotion()
        session_tip    = SESSION_SUGGESTIONS.get(dominant, "")
        return {
            "duration_seconds": total_seconds,
            "overall_focus":    round(self.focus_score, 1),
            "emotion_times":    emotion_times,
            "alert_count":      len(self.alert_log),
            "alerts":           self.alert_log,
            "dominant_emotion": dominant,
            "session_tip":      session_tip,
        }