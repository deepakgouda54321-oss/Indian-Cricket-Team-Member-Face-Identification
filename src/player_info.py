"""
Information and metadata for Indian Cricket Team members.
"""

PLAYERS_INFO = {
    "virat_kohli": {
        "display_name": "Virat Kohli",
        "nickname": "King Kohli / Run Machine",
        "jersey_number": 18,
        "role": "Top-Order Batter (Right-hand bat)",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm medium",
        "notable_achievements": [
            "ICC Cricketer of the Decade (2011-2020)",
            "50+ ODI Centuries (Highest in cricket history)",
            "ICC Cricket World Cup Champion (2011), Champions Trophy (2013), T20 World Cup (2024)",
            "Over 26,000 international runs across all formats"
        ],
        "facial_features_note": "Sharp jawline, well-groomed beard, intense focused gaze, styled athletic haircut."
    },
    "rohit_sharma": {
        "display_name": "Rohit Sharma",
        "nickname": "Hitman",
        "jersey_number": 45,
        "role": "Opening Batter & Indian Captain (Right-hand bat)",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm off break",
        "notable_achievements": [
            "ICC T20 World Cup Winning Captain (2024)",
            "Only player with 3 Double Centuries in ODI cricket (record 264 vs Sri Lanka)",
            "Highest individual score in ODI history (264)",
            "5 IPL titles as Captain for Mumbai Indians"
        ],
        "facial_features_note": "Rounder face structure, calm expression, neatly trimmed beard and mustache, classic side-part hair."
    },
    "ms_dhoni": {
        "display_name": "MS Dhoni",
        "nickname": "Captain Cool / Thala / MSD",
        "jersey_number": 7,
        "role": "Wicketkeeper-Batter & Legendary Former Captain",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm medium",
        "notable_achievements": [
            "Only Captain to win all 3 major ICC Trophies (T20 WC 2007, ODI WC 2011, CT 2013)",
            "Iconic World Cup winning 6 at Wankhede Stadium (2011)",
            "Over 10,000 ODI runs with an average over 50 as a finisher",
            "5 IPL titles as Captain for Chennai Super Kings"
        ],
        "facial_features_note": "Strong masculine jaw, calm and steady eyes, light salt-and-pepper stubble, confident demeanor."
    },
    "sachin_tendulkar": {
        "display_name": "Sachin Tendulkar",
        "nickname": "Master Blaster / God of Cricket",
        "jersey_number": 10,
        "role": "Top-Order Batter (Right-hand bat)",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm leg spin / off break",
        "notable_achievements": [
            "100 International Centuries (World Record)",
            "Highest run-scorer in international cricket (34,357 runs)",
            "First player to score a double century in Men's ODIs (200* vs South Africa)",
            "2011 ICC Cricket World Cup Champion, Bharat Ratna Awardee"
        ],
        "facial_features_note": "Signature curly hair, compact facial profile, warm gentle smile, iconic expressive eyes."
    },
    "hardik_pandya": {
        "display_name": "Hardik Pandya",
        "nickname": "Kung Fu Pandya",
        "jersey_number": 33,
        "role": "All-Rounder & Dynamic Finisher",
        "batting_style": "Right-handed explosive bat",
        "bowling_style": "Right-arm fast-medium",
        "notable_achievements": [
            "Bowled the historic final over in ICC T20 World Cup 2024 final victory",
            "IPL 2022 Champion as Captain of Gujarat Titans",
            "Premier high-impact seam bowling all-rounder for India",
            "Multiple match-winning knocks under pressure"
        ],
        "facial_features_note": "Sharp angular cheekbones, sculpted designer beard/goatee, trendy modern fade haircut."
    },
    "jasprit_bumrah": {
        "display_name": "Jasprit Bumrah",
        "nickname": "Boom Boom / Yorker King",
        "jersey_number": 93,
        "role": "Premier Fast Bowler",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm fast (unorthodox lethal action)",
        "notable_achievements": [
            "Player of the Tournament in ICC T20 World Cup 2024",
            "Ranked ICC World No. 1 bowler across formats",
            "Fastest Indian pacer to 150 Test wickets",
            "Unrivaled pinpoint yorkers and deceptive slower balls"
        ],
        "facial_features_note": "Bright welcoming smile, distinctive smile lines, naturally curly hair, light stubble."
    }
}


def get_player_info(player_key: str) -> dict:
    """Retrieve player metadata with clean fallback."""
    key = player_key.lower().strip().replace(" ", "_")
    return PLAYERS_INFO.get(key, {
        "display_name": player_key.replace("_", " ").title(),
        "nickname": "Indian Cricket Team Star",
        "jersey_number": "N/A",
        "role": "Cricket Player",
        "batting_style": "N/A",
        "bowling_style": "N/A",
        "notable_achievements": ["Indian National Cricket Team Member"],
        "facial_features_note": "Facial features extracted via Computer Vision pipeline."
    })

