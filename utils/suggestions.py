"""
================================================================================
 AI-Based Digital Burnout & Cognitive Load Detection System
 File: utils/suggestions.py
 Purpose: Rule-based smart recommendation engine
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Suggestion:
    icon: str
    category: str
    title: str
    detail: str
    priority: int   # 1 = high, 2 = medium, 3 = low


def generate_suggestions(raw: dict, prediction: dict) -> List[Dict]:
    """
    Generate personalised, actionable recommendations based on both the
    raw input values and the model's predictions.
    Returns a list of suggestion dicts sorted by priority.
    """
    suggestions: List[Suggestion] = []
    burnout = prediction["burnout_int"]          # 0/1/2
    cog_load = prediction["cognitive_load"]
    distraction = prediction["distraction_index"]
    productivity = prediction["productivity"]

    # ── Sleep ────────────────────────────────────────────────────────────────
    if raw["sleep_hours"] < 6:
        suggestions.append(Suggestion(
            icon="🌙", category="Sleep",
            title="Critical Sleep Deficit Detected",
            detail=f"You're averaging {raw['sleep_hours']:.1f}h of sleep. "
                   "Research shows <6h impairs memory consolidation by up to 40%. "
                   "Aim for 7–9 hours with a consistent bedtime.",
            priority=1
        ))
    elif raw["sleep_hours"] < 7:
        suggestions.append(Suggestion(
            icon="💤", category="Sleep",
            title="Improve Sleep Duration",
            detail="You're slightly below the 7–9h recommendation. "
                   "Try a 30-min earlier bedtime and avoid screens 1h before sleep.",
            priority=2
        ))

    # ── Screen time ──────────────────────────────────────────────────────────
    if raw["screen_time_hours"] > 10:
        suggestions.append(Suggestion(
            icon="📵", category="Screen Time",
            title="Excessive Screen Exposure",
            detail=f"{raw['screen_time_hours']:.1f}h of screen time significantly "
                   "elevates eye strain and cortisol levels. Use the 20-20-20 rule: "
                   "every 20 min, look at something 20 ft away for 20 seconds.",
            priority=1
        ))
    elif raw["screen_time_hours"] > 7:
        suggestions.append(Suggestion(
            icon="👁️", category="Screen Time",
            title="Moderate Your Screen Time",
            detail="Consider scheduling screen-free blocks of at least 30 minutes. "
                   "Enable blue-light filters after 6 PM.",
            priority=2
        ))

    # ── Breaks ───────────────────────────────────────────────────────────────
    if raw["break_frequency"] < 3:
        suggestions.append(Suggestion(
            icon="⏸️", category="Breaks",
            title="Insufficient Breaks",
            detail="Fewer than 3 breaks per day is associated with 60% higher "
                   "error rates. Try the Pomodoro Technique: 25 min work + 5 min break.",
            priority=1
        ))

    # ── Typing / Cognitive signals ───────────────────────────────────────────
    if raw["backspace_rate"] > 15:
        suggestions.append(Suggestion(
            icon="⌨️", category="Cognitive Load",
            title="High Error Rate in Typing",
            detail=f"A backspace rate of {raw['backspace_rate']:.1f}% suggests "
                   "cognitive overload or fatigue. Short 5-min mental breaks can "
                   "restore working memory capacity.",
            priority=2
        ))

    if cog_load > 65:
        suggestions.append(Suggestion(
            icon="🧠", category="Cognitive Load",
            title="Cognitive Overload Warning",
            detail=f"Your cognitive load score is {cog_load:.0f}/100. "
                   "Offload tasks by writing to-do lists, using templates, and "
                   "chunking complex work into smaller tasks.",
            priority=1 if cog_load > 80 else 2
        ))

    # ── Distraction ──────────────────────────────────────────────────────────
    if distraction > 50:
        suggestions.append(Suggestion(
            icon="🎯", category="Focus",
            title="High Distraction Detected",
            detail=f"App-switch rate of {raw['app_switch_rate']:.0f}/hr is well above "
                   "optimal (<15/hr). Enable Do Not Disturb mode and use site blockers "
                   "during deep work sessions.",
            priority=1
        ))
    elif distraction > 30:
        suggestions.append(Suggestion(
            icon="📱", category="Focus",
            title="Moderate Distraction Level",
            detail="Try batching notifications and checking messages only at "
                   "scheduled intervals (e.g. every 2 hours).",
            priority=2
        ))

    # ── Social media usage ───────────────────────────────────────────────────
    if raw["usage_type"] == 1:
        suggestions.append(Suggestion(
            icon="📲", category="App Usage",
            title="Heavy Social Media Usage",
            detail="Social media-dominant usage is linked to increased anxiety and "
                   "reduced deep-focus capacity. Set a daily screen-time budget "
                   "using your device's built-in tools.",
            priority=2
        ))

    # ── Physical activity ────────────────────────────────────────────────────
    if raw["physical_activity"] < 20:
        suggestions.append(Suggestion(
            icon="🏃", category="Wellness",
            title="Increase Physical Activity",
            detail=f"Only {raw['physical_activity']:.0f} min of activity today. "
                   "Even a 20-min walk raises BDNF (brain-derived neurotrophic factor) "
                   "and improves cognitive performance by ~15%.",
            priority=2
        ))

    # ── Hydration ────────────────────────────────────────────────────────────
    if raw["hydration_level"] <= 2:
        suggestions.append(Suggestion(
            icon="💧", category="Wellness",
            title="Dehydration Risk",
            detail="Low hydration is one of the most overlooked cognitive impairments. "
                   "A 2% drop in body water reduces concentration by ~13%. "
                   "Aim for 8 glasses (2L) per day.",
            priority=2
        ))

    # ── Burnout-specific ─────────────────────────────────────────────────────
    if burnout == 2:
        suggestions.append(Suggestion(
            icon="🚨", category="Burnout Recovery",
            title="High Burnout — Immediate Action Needed",
            detail="Your profile indicates high burnout risk. Consider a structured "
                   "digital detox weekend, discuss workload redistribution with peers, "
                   "and consult a mental wellness resource.",
            priority=1
        ))
        suggestions.append(Suggestion(
            icon="🧘", category="Mindfulness",
            title="Practice Mindful Recovery",
            detail="5-minute mindfulness or box-breathing exercises (4s in, 4s hold, "
                   "4s out, 4s hold) significantly reduce cortisol within a single session.",
            priority=1
        ))
    elif burnout == 1:
        suggestions.append(Suggestion(
            icon="⚠️", category="Prevention",
            title="Burnout Risk Building Up",
            detail="You're in the medium-risk zone. Establish a structured daily routine "
                   "with explicit off-screen time before this escalates.",
            priority=2
        ))
    else:
        suggestions.append(Suggestion(
            icon="✅", category="Maintenance",
            title="Healthy Digital Habits — Keep It Up",
            detail="Your current habits show low burnout risk. "
                   "Maintain consistent sleep, regular breaks, and balanced app usage "
                   "to preserve this state.",
            priority=3
        ))

    # ── Productivity boost ───────────────────────────────────────────────────
    if productivity < 50:
        suggestions.append(Suggestion(
            icon="⚡", category="Productivity",
            title="Low Productivity Score",
            detail=f"Your productivity score is {productivity:.0f}/100. "
                   "Start the day with your 3 most important tasks (MIT framework) "
                   "before checking email or social media.",
            priority=2
        ))

    # Sort: high priority first, then alphabetically by category
    suggestions.sort(key=lambda s: (s.priority, s.category))

    return [
        {
            "icon":     s.icon,
            "category": s.category,
            "title":    s.title,
            "detail":   s.detail,
            "priority": s.priority,
        }
        for s in suggestions
    ]


def generate_ai_report(raw: dict, prediction: dict, suggestions: list) -> str:
    """
    Generate a structured daily AI report summarising the user's digital health.
    Returns a plain-text / HTML-friendly report string.
    """
    level = prediction["burnout_level"]
    cog   = prediction["cognitive_load"]
    prod  = prediction["productivity"]
    dist  = prediction["distraction_index"]

    top_issues = [s["title"] for s in suggestions if s["priority"] == 1][:3]
    issue_str  = "; ".join(top_issues) if top_issues else "No critical issues identified"

    usage_labels = {0: "Study-focused", 1: "Social media-heavy", 2: "Mixed"}
    usage_label  = usage_labels.get(raw["usage_type"], "Unknown")

    report = f"""
    ╔══════════════════════════════════════════════════════╗
    ║         DAILY DIGITAL WELLBEING REPORT               ║
    ╚══════════════════════════════════════════════════════╝

    BURNOUT LEVEL   : {level}
    COGNITIVE LOAD  : {cog:.1f} / 100
    PRODUCTIVITY    : {prod:.1f} / 100
    DISTRACTION IDX : {dist:.1f} / 100

    ── Key Inputs ──────────────────────────────────────────
    Screen Time     : {raw['screen_time_hours']:.1f} hours
    Sleep           : {raw['sleep_hours']:.1f} hours
    Work Hours      : {raw['work_hours']:.1f} hours
    Typing Speed    : {raw['typing_speed_wpm']:.0f} WPM
    App Usage Type  : {usage_label}
    Breaks Taken    : {raw['break_frequency']}
    Physical Act.   : {raw['physical_activity']:.0f} min

    ── Critical Issues ─────────────────────────────────────
    {issue_str}

    ── Top Recommendation ──────────────────────────────────
    {suggestions[0]['title'] if suggestions else 'N/A'}
    {suggestions[0]['detail'] if suggestions else ''}

    ── Model Confidence ────────────────────────────────────
    {' | '.join(f"{k}: {v:.1f}%" for k, v in prediction['probabilities'].items())}
    Model Accuracy: {prediction['model_accuracy'] * 100:.2f}%
    """
    return report.strip()
