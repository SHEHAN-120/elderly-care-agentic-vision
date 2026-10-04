"""
Agentic reasoning: when classifier confidence is low, look at temporal context
(previous + following segments) to decide the correct state.
"""
from typing import List, Optional
from backend.core.config import State, SystemConfig


class ContextAgent:
    def __init__(self, config: SystemConfig):
        self.cfg = config
        self.trace: List[str] = []

    def resolve(
        self,
        time_sec: float,
        raw_state: str,
        confidence: float,
        recent_history: List["object"],   # list of (time_sec, state, conf)
    ) -> tuple:
        """
        Returns (final_state, final_conf, reasoning_string).
        """
        if confidence >= self.cfg.agent_confidence_threshold:
            return raw_state, confidence, ""

        # Gather previous context within window
        window = self.cfg.agent_context_window_sec
        prev = [h for h in recent_history if 0 < (time_sec - h[0]) <= window]

        reasoning = []
        reasoning.append(f"[OBSERVATION] t={time_sec:.2f}s raw={raw_state} conf={confidence:.2f}")
        reasoning.append(f"[ACTION] confidence below {self.cfg.agent_confidence_threshold} → gathering context")

        if prev:
            prev_states = [p[1] for p in prev]
            reasoning.append(f"[FINDING] previous states in last {window}s: {prev_states}")

            most_common = max(set(prev_states), key=prev_states.count)

            # Rule 1: was in bed, now upright/moving → possible bed exit transition
            if most_common in (State.LYING_IN_BED, State.SITTING_ON_BED):
                if raw_state in (State.STANDING, State.WALKING, State.SITTING_OUTSIDE_BED):
                    reasoning.append("[CONCLUSION] transition from bed detected → trust raw state")
                    return raw_state, max(confidence, 0.70), " | ".join(reasoning)
                if raw_state == State.OUT_OF_BED and confidence < 0.55:
                    reasoning.append("[CONCLUSION] low-conf OUT_OF_BED while previously in bed → SITTING_ON_BED")
                    return State.SITTING_ON_BED, 0.60, " | ".join(reasoning)

            # Rule 2: unknown raw → fall back to previous stable state
            if raw_state == State.UNKNOWN:
                reasoning.append(f"[CONCLUSION] insufficient evidence → inherit '{most_common}'")
                return most_common, max(confidence, 0.55), " | ".join(reasoning)

            # Rule 3: ambiguous nearby activity → blend
            reasoning.append(f"[CONCLUSION] contextual prior '{most_common}' agrees with raw → accept")
            return raw_state, max(confidence, 0.65), " | ".join(reasoning)

        # No context → keep UNKNOWN
        reasoning.append("[CONCLUSION] no usable context → UNKNOWN")
        return State.UNKNOWN, confidence, " | ".join(reasoning)