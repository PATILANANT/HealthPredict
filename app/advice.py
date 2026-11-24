"""Plain-language next steps shown on the result page and in the report."""

ADVICE = {
    "diabetes": {
        "High": ["Book an appointment with a doctor and ask for an HbA1c or fasting glucose test.",
                 "Cut down on sugary drinks, sweets and refined carbs (white rice, maida).",
                 "Aim for 30 minutes of brisk walking at least 5 days a week."],
        "Moderate": ["Get a routine blood-sugar check in the next few months.",
                     "Keep a healthy weight: even 5% weight loss lowers diabetes risk.",
                     "Prefer whole grains, vegetables and pulses; limit fried food."],
        "Low": ["Keep up an active lifestyle and balanced diet.",
                "Repeat a blood-sugar check once a year, especially after age 35."],
    },
    "stroke": {
        "High": ["Please see a doctor soon to review blood pressure, sugar and cholesterol.",
                 "Learn the warning signs - BE FAST: Balance, Eyes, Face drooping, Arm weakness, "
                 "Speech difficulty, Time to call 108/112.",
                 "If you smoke, quitting is the single biggest step you can take."],
        "Moderate": ["Check your blood pressure regularly and keep it under 130/80.",
                     "Stay active, reduce salt and keep a healthy weight.",
                     "Discuss your risk factors at your next check-up."],
        "Low": ["Keep a healthy routine: regular activity, low salt, no smoking.",
                "Have your blood pressure checked at least once a year."],
    },
    "mental_health": {
        "High": ["Please talk to someone you trust today, or call Tele-MANAS 14416 (free, 24x7).",
                 "Consider booking an appointment with a psychologist or psychiatrist.",
                 "You do not have to handle this alone - reaching out is a sign of strength."],
        "Moderate": ["Talking to a counsellor can help you manage stress and worry.",
                     "Protect your sleep, move your body daily and take short breaks.",
                     "Share how you feel with a friend or family member."],
        "Low": ["Keep doing what helps you feel well - sleep, exercise, time with people.",
                "Check in with yourself regularly; screening again later is easy."],
    },
}


def advice_for(module: str, risk_level: str) -> list[str]:
    return ADVICE.get(module, {}).get(risk_level, [])
