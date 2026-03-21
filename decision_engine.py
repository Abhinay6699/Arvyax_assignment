"""
decision_engine.py — Rule-based logic for WHAT to do and WHEN to do it.
Exhaustive coverage of all emotional states, time-of-day, and intensity combinations.
"""


def decide(predicted_state, intensity, stress_level, energy_level, time_of_day):
    """
    Determine what action to take and when, based on predicted emotional state,
    intensity, stress/energy levels, and time of day.

    Args:
        predicted_state (str): Predicted emotional state (e.g., 'anxious', 'calm', 'sad', etc.)
        intensity (int): Predicted intensity (1-5)
        stress_level (float): Stress level from input data
        energy_level (float): Energy level from input data
        time_of_day (str): Time of day ('morning', 'afternoon', 'evening', 'night')

    Returns:
        tuple: (what_to_do, when_to_do)
    """
    predicted_state = str(predicted_state).lower().strip()
    time_of_day = str(time_of_day).lower().strip()
    intensity = int(round(float(intensity)))
    stress_level = float(stress_level) if stress_level is not None else 2.0
    energy_level = float(energy_level) if energy_level is not None else 2.0

    # Clamp values
    intensity = max(1, min(5, intensity))

    # ===========================
    # WHAT TO DO — Priority Rules
    # ===========================
    what_to_do = "pause"  # default
    
    # 1. Box Breathing: Extreme states (intensity >= 4) OR Extreme Stress (>= 5)
    if (predicted_state in ["overwhelmed", "anxious", "panicked"] and intensity >= 4) or stress_level >= 5:
        what_to_do = "box_breathing"

    # 2. Deep Work: Focused/Calm with moderate energy (>= 3) and NOT intense overwhelmed
    elif predicted_state in ["focused", "motivated"] and energy_level >= 3 and time_of_day in ["morning", "afternoon"]:
        what_to_do = "deep_work"

    # 3. Yoga: Calm/Peaceful/Relaxed (Morning/Afternoon) OR Mixed (Low intensity)
    elif (predicted_state in ["calm", "peaceful", "relaxed"] and time_of_day in ["morning", "afternoon"]) or (predicted_state == "mixed" and intensity <= 2):
        what_to_do = "yoga"

    # 4. Movement: Happy/Focused with high energy (>= 4)
    elif predicted_state in ["happy", "focused", "energetic"] and energy_level >= 4:
        what_to_do = "movement"

    # 5. Grounding: Overwhelmed (int 3) OR Restless (int >= 2) OR Moderate-high Stress
    elif (predicted_state == "overwhelmed" and intensity == 3) or (predicted_state in ["restless", "distracted"] and intensity >= 2) or (stress_level >= 3 and energy_level >= 4):
        what_to_do = "grounding"

    # 6. Journaling: Sad/Mixed states OR Neutral/Mixed (Evening/Night)
    elif predicted_state in ["sad", "mixed", "melancholic", "down"] or (predicted_state in ["neutral", "mixed"] and time_of_day in ["evening", "night"]):
        what_to_do = "journaling"

    # 7. Sound Therapy: Overwhelmed (int <= 2) OR Neutral (Evening)
    elif (predicted_state == "overwhelmed" and intensity <= 2) or (predicted_state == "neutral" and time_of_day == "evening"):
        what_to_do = "sound_therapy"

    # 8. Rest: Low Energy (<= 2) OR Night time OR Overwhelmed at Night
    elif energy_level <= 2 or time_of_day == "night":
        what_to_do = "rest"

    # 9. Light Planning: Neutral/Focused with moderate energy (2-3)
    elif predicted_state in ["neutral", "focused", "content", "calm"] and 2 <= energy_level <= 3:
        what_to_do = "light_planning"

    # 10. Secondary mapping to eliminate 'pause'
    elif predicted_state in ["happy", "excited"]:
        what_to_do = "movement"
    elif predicted_state in ["calm", "peaceful", "relaxed"]:
        what_to_do = "yoga"
    elif predicted_state in ["sad", "low"]:
        what_to_do = "journaling"
    elif predicted_state in ["anxious", "restless", "overwhelmed"]:
        what_to_do = "grounding"
    elif time_of_day == "morning":
        what_to_do = "light_planning"
    else:
        what_to_do = "grounding"

    # ===========================
    # WHEN TO DO — Urgency Rules
    # ===========================
    when_to_do = "within_15_min"  # default

    # Rule 1: Critical urgency — high intensity AND high stress
    if intensity >= 4 and stress_level >= 4:
        when_to_do = "now"

    # Rule 2: High intensity alone
    elif intensity >= 4:
        when_to_do = "now"

    # Rule 3: Moderate intensity
    elif intensity >= 3:
        when_to_do = "within_15_min"

    # Rule 4: Morning calm/happy — can schedule for later
    elif time_of_day == "morning" and predicted_state in ["calm", "happy", "content", "peaceful", "joyful", "relaxed"]:
        when_to_do = "later_today"

    # Rule 5: Evening time
    elif time_of_day in ["evening", "night"]:
        when_to_do = "tonight"

    # Rule 6: Low urgency — low intensity and low stress
    elif intensity <= 2 and stress_level <= 2:
        when_to_do = "tomorrow_morning"

    # Rule 7: Afternoon with moderate signals
    elif time_of_day == "afternoon" and intensity <= 2:
        when_to_do = "later_today"

    # Default
    else:
        when_to_do = "within_15_min"

    return what_to_do, when_to_do


def decide_batch(df, state_col='predicted_state', intensity_col='predicted_intensity'):
    """
    Apply decision engine to an entire DataFrame.
    Returns lists of what_to_do and when_to_do.
    """
    what_list = []
    when_list = []

    for _, row in df.iterrows():
        state = row.get(state_col, "neutral")
        intensity = row.get(intensity_col, 3)
        stress = row.get('stress_level', 2)
        energy = row.get('energy_level', 2)
        time = row.get('time_of_day', 'morning')

        what, when = decide(state, intensity, stress, energy, time)
        what_list.append(what)
        when_list.append(when)

    return what_list, when_list


if __name__ == "__main__":
    # Test with sample inputs
    test_cases = [
        ("anxious", 4, 4, 3, "morning"),
        ("calm", 2, 1, 4, "morning"),
        ("sad", 4, 3, 2, "evening"),
        ("happy", 2, 1, 4, "afternoon"),
        ("restless", 1, 2, 3, "night"),
        ("neutral", 3, 3, 3, "afternoon"),
    ]

    print("Decision Engine Test Cases:")
    print(f"  {'State':<12} {'Int':>3} {'Str':>3} {'Eng':>3} {'Time':<12} → {'What':<18} {'When':<18}")
    print("  " + "─" * 80)
    for state, intensity, stress, energy, time in test_cases:
        what, when = decide(state, intensity, stress, energy, time)
        print(f"  {state:<12} {intensity:>3} {stress:>3} {energy:>3} {time:<12} → {what:<18} {when:<18}")
