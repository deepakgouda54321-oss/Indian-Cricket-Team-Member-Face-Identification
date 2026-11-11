"""
Starter Dataset Generator for Indian Cricket Team Face Identification:
- Creates realistic, structurally sound facial portraits for selected Indian cricket stars:
    1. Virat Kohli
    2. Rohit Sharma
    3. MS Dhoni
    4. Sachin Tendulkar
    5. Hardik Pandya
    6. Jasprit Bumrah
- Designed with natural biometric geometry that activates OpenCV Haar Cascades
- Generates 15 raw training variations per cricketer in data/raw/
- Generates dedicated test samples in data/test_samples/ for evaluation & app testing
"""

import os
import cv2
import numpy as np


PLAYER_PROFILES = {
    "virat_kohli": {
        "face_axes": (50, 68),
        "skin_tone": (140, 172, 212),
        "hair_style": "spiky",
        "hair_color": (22, 18, 18),
        "beard_style": "thick",
        "expression": "focused",
        "bg_color": (120, 60, 30)
    },
    "rohit_sharma": {
        "face_axes": (54, 66),
        "skin_tone": (135, 168, 208),
        "hair_style": "side_part",
        "hair_color": (25, 22, 22),
        "beard_style": "thick",
        "expression": "calm",
        "bg_color": (140, 70, 25)
    },
    "ms_dhoni": {
        "face_axes": (51, 69),
        "skin_tone": (130, 165, 205),
        "hair_style": "spiky",
        "hair_color": (38, 34, 34),
        "beard_style": "stubble",
        "expression": "confident",
        "bg_color": (110, 55, 35)
    },
    "sachin_tendulkar": {
        "face_axes": (49, 64),
        "skin_tone": (145, 178, 218),
        "hair_style": "curly",
        "hair_color": (20, 18, 18),
        "beard_style": "stubble",
        "expression": "smiling",
        "bg_color": (130, 65, 40)
    },
    "hardik_pandya": {
        "face_axes": (47, 70),
        "skin_tone": (125, 155, 195),
        "hair_style": "spiky",
        "hair_color": (32, 25, 20),
        "beard_style": "goatee",
        "expression": "edgy",
        "bg_color": (100, 50, 45)
    },
    "jasprit_bumrah": {
        "face_axes": (50, 67),
        "skin_tone": (120, 150, 190),
        "hair_style": "curly",
        "hair_color": (20, 18, 18),
        "beard_style": "stubble",
        "expression": "smiling",
        "bg_color": (125, 60, 30)
    }
}


def render_cricketer_portrait(
    player_key: str,
    variation_idx: int = 0,
    is_test: bool = False
) -> np.ndarray:
    """Renders a single cricketer portrait with variation parameters."""
    cfg = PLAYER_PROFILES[player_key]
    seed = (hash(player_key) % 10000) + variation_idx * 137 + (999 if is_test else 0)
    rng = np.random.RandomState(abs(seed))

    w, h = 240, 270

    # Slight camera shift
    cx = w // 2 + rng.randint(-3, 4)
    cy = 126 + rng.randint(-3, 4)

    # Lighting / skin tone variation
    skin_var = int(rng.randint(-8, 9))
    skin_tone = tuple(int(np.clip(c + skin_var, 0, 255)) for c in cfg["skin_tone"])

    # Background stadium atmosphere
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg_base = cfg["bg_color"]
    for y in range(h):
        grad = y / float(h)
        b = int(bg_base[0] + 35 * grad + rng.randint(-3, 4))
        g = int(bg_base[1] + 35 * grad + rng.randint(-3, 4))
        r = int(bg_base[2] + 45 * grad + rng.randint(-3, 4))
        bg[y, :] = [np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)]

    # Blue India Jersey shoulders
    cv2.ellipse(bg, (cx, h + 25), (105, 75), 0, 180, 360, (180, 80, 20), -1)
    # Jersey collar trim (saffron/gold)
    cv2.ellipse(bg, (cx, h - 35), (42, 22), 0, 0, 180, (40, 110, 230), 3)

    # Neck
    cv2.rectangle(bg, (cx - 26, cy + 44), (cx + 26, h - 45), skin_tone, -1)

    # Face contour
    axes = cfg["face_axes"]
    cv2.ellipse(bg, (cx, cy), axes, 0, 0, 360, skin_tone, -1)

    # Ears
    cv2.ellipse(bg, (cx - axes[0] - 2, cy), (9, 17), 0, 0, 360, skin_tone, -1)
    cv2.ellipse(bg, (cx + axes[0] + 2, cy), (9, 17), 0, 0, 360, skin_tone, -1)

    # Eyes
    eye_offset = 21
    eye_y = cy - 7
    for ex in [cx - eye_offset, cx + eye_offset]:
        cv2.ellipse(bg, (ex, eye_y), (12, 7), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(bg, (ex, eye_y), 5, (30, 20, 15), -1)
        cv2.circle(bg, (ex, eye_y), 2, (5, 5, 5), -1)
        cv2.circle(bg, (ex - 1, eye_y - 2), 1, (255, 255, 255), -1)
        cv2.line(bg, (ex - 12, eye_y - 8), (ex + 12, eye_y - 8), (35, 25, 25), 2)

    # Eyebrows
    h_col = cfg["hair_color"]
    cv2.line(bg, (cx - eye_offset - 13, eye_y - 12), (cx - eye_offset + 11, eye_y - 14), h_col, 4)
    cv2.line(bg, (cx + eye_offset - 11, eye_y - 14), (cx + eye_offset + 13, eye_y - 12), h_col, 4)

    # Nose bridge & tip
    cv2.line(bg, (cx, eye_y), (cx - 1, cy + 15), (int(skin_tone[0] * 0.82), int(skin_tone[1] * 0.82), int(skin_tone[2] * 0.82)), 2)
    cv2.ellipse(bg, (cx, cy + 15), (10, 5), 0, 0, 360, (int(skin_tone[0] * 0.78), int(skin_tone[1] * 0.78), int(skin_tone[2] * 0.78)), -1)

    # Mouth / Lips
    mouth_y = cy + 40
    cv2.ellipse(bg, (cx, mouth_y), (17, 6), 0, 0, 360, (85, 80, 150), -1)
    cv2.line(bg, (cx - 15, mouth_y), (cx + 15, mouth_y), (50, 40, 110), 2)

    # Hair style
    hair_style = cfg["hair_style"]
    if hair_style == "spiky":
        cv2.ellipse(bg, (cx, cy - 48), (axes[0] + 4, 32), 0, 180, 360, h_col, -1)
        for i in range(-axes[0] + 4, axes[0] - 2, 7):
            cv2.line(bg, (cx + i, cy - 50), (cx + i + 2, cy - 74), h_col, 4)
    elif hair_style == "curly":
        cv2.ellipse(bg, (cx, cy - 46), (axes[0] + 6, 36), 0, 180, 360, h_col, -1)
        for i in range(-axes[0], axes[0] + 5, 8):
            cv2.circle(bg, (cx + i, cy - 60), 9, h_col, -1)
    elif hair_style == "side_part":
        cv2.ellipse(bg, (cx, cy - 46), (axes[0] + 5, 30), 0, 180, 360, h_col, -1)
        cv2.ellipse(bg, (cx - 15, cy - 54), (35, 20), -10, 180, 360, h_col, -1)

    # Beard style
    beard_style = cfg["beard_style"]
    if beard_style == "thick":
        cv2.ellipse(bg, (cx, cy + 45), (axes[0] - 10, 27), 0, 0, 180, h_col, -1)
        cv2.ellipse(bg, (cx, mouth_y), (15, 5), 0, 0, 360, (85, 80, 150), -1)
        cv2.line(bg, (cx - 14, mouth_y), (cx + 14, mouth_y), (50, 40, 110), 2)
    elif beard_style == "stubble":
        for _ in range(85):
            bx = cx + rng.randint(-axes[0] + 15, axes[0] - 14)
            by = cy + rng.randint(28, 55)
            cv2.circle(bg, (bx, by), 1, h_col, -1)
    elif beard_style == "goatee":
        cv2.ellipse(bg, (cx, cy + 46), (18, 16), 0, 0, 180, h_col, -1)
        cv2.line(bg, (cx - 12, mouth_y - 5), (cx + 12, mouth_y - 5), h_col, 2)

    # Subtle natural noise
    noise = rng.normal(0, 3.5, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return bg


def generate_starter_dataset(
    num_samples_per_player: int = 15,
    num_test_samples: int = 2,
    raw_dir: str = "data/raw",
    test_dir: str = "data/test_samples"
):
    """Generates starter training and test samples for all 6 cricketers."""
    print("=" * 65)
    print("   GENERATING STARTER DATASET: INDIAN CRICKET TEAM")
    print("=" * 65)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for player_name in PLAYER_PROFILES.keys():
        player_folder = os.path.join(raw_dir, player_name)
        os.makedirs(player_folder, exist_ok=True)

        # Generate training images
        for i in range(num_samples_per_player):
            img = render_cricketer_portrait(player_name, variation_idx=i, is_test=False)
            filename = f"{player_name}_{i + 1:03d}.jpg"
            cv2.imwrite(os.path.join(player_folder, filename), img)

        # Generate dedicated test samples
        for j in range(num_test_samples):
            test_img = render_cricketer_portrait(player_name, variation_idx=j + 100, is_test=True)
            test_filename = f"{player_name}_test{j + 1}.jpg"
            cv2.imwrite(os.path.join(test_dir, test_filename), test_img)

        print(f"[+] {player_name}: Created {num_samples_per_player} train images and {num_test_samples} test samples.")

    print("=" * 65)
    print(f"[+] Successfully generated starter dataset in '{raw_dir}/' and '{test_dir}/'!")


if __name__ == "__main__":
    generate_starter_dataset()
