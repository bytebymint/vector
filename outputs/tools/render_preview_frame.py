import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_fonts():
    font_dir = r'C:\Windows\Fonts'
    try:
        f_title = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 18)
        f_body = ImageFont.truetype(os.path.join(font_dir, 'segoeui.ttf'), 13)
        f_body_b = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 13)
        f_small = ImageFont.truetype(os.path.join(font_dir, 'segoeui.ttf'), 11)
        f_small_b = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 11)
        f_mono = ImageFont.truetype(os.path.join(font_dir, 'consola.ttf'), 12)
        f_badge = ImageFont.truetype(os.path.join(font_dir, 'segoeuib.ttf'), 10)
    except Exception:
        f_title = f_body = f_body_b = f_small = f_small_b = f_mono = f_badge = ImageFont.load_default()
    return {
        'title': f_title, 'body': f_body, 'body_b': f_body_b,
        'small': f_small, 'small_b': f_small_b, 'mono': f_mono, 'badge': f_badge
    }

def draw_rounded_rect(draw, bbox, radius, fill=None, outline=None, width=1):
    x1, y1, x2, y2 = bbox
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)

def render_composite_frame(video_frame, current_sec, profiles_data, active_faces, fonts):
    # Canvas: 1920 x 1080
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Background: #080e16
    canvas[:] = (22, 14, 8) # BGR
    
    # 1. TOPBAR (0 to 60)
    cv2.rectangle(canvas, (0, 0), (1920, 56), (29, 19, 11), -1)
    cv2.line(canvas, (0, 56), (1920, 56), (62, 48, 34), 1)

    # 2. WORKSPACE LAYOUT
    # Left streaming window: x: 18 to 1180 (width: 1162)
    # Right profiles gallery: x: 1200 to 1902 (width: 702)
    left_x1, left_y1, left_x2, left_y2 = 18, 68, 1182, 1032
    right_x1, right_y1, right_x2, right_y2 = 1200, 68, 1902, 1032

    # Draw panel backgrounds with sleek borders
    cv2.rectangle(canvas, (left_x1, left_y1), (left_x2, left_y2), (35, 25, 16), -1)
    cv2.rectangle(canvas, (left_x1, left_y1), (left_x2, left_y2), (62, 48, 34), 1)
    cv2.rectangle(canvas, (left_x1, left_y1), (left_x2, left_y1 + 42), (40, 28, 18), -1)
    cv2.line(canvas, (left_x1, left_y1 + 42), (left_x2, left_y1 + 42), (62, 48, 34), 1)

    cv2.rectangle(canvas, (right_x1, right_y1), (right_x2, right_y2), (35, 25, 16), -1)
    cv2.rectangle(canvas, (right_x1, right_y1), (right_x2, right_y2), (62, 48, 34), 1)
    cv2.rectangle(canvas, (right_x1, right_y1), (right_x2, right_y1 + 42), (40, 28, 18), -1)
    cv2.line(canvas, (right_x1, right_y1 + 42), (right_x2, right_y1 + 42), (62, 48, 34), 1)

    # Draw Video onto left panel
    # Video target area: x: 28 to 1172 (width: 1144), y: 120 to 764 (height: 644)
    vw, vh = 1144, 644
    vx1, vy1 = 28, 120
    resized_video = cv2.resize(video_frame, (vw, vh))

    # Draw active faces on video before placing
    for face in active_faces:
        # face bbox in video frame coordinates
        bx1, by1, bw, bh = face['box']
        # scale to resized video
        sx = vw / video_frame.shape[1]
        sy = vh / video_frame.shape[0]
        fx1 = int(bx1 * sx)
        fy1 = int(by1 * sy)
        fx2 = int((bx1 + bw) * sx)
        fy2 = int((by1 + bh) * sy)

        # Draw lock-on bounding box (cyan / emerald)
        color = (208, 217, 54) # BGR for #36d9d0
        cv2.rectangle(resized_video, (fx1, fy1), (fx2, fy2), color, 2)
        # Lock brackets at corners
        c_len = min(14, bw // 4, bh // 4)
        cv2.line(resized_video, (fx1, fy1), (fx1 + c_len, fy1), (255, 255, 255), 2)
        cv2.line(resized_video, (fx1, fy1), (fx1, fy1 + c_len), (255, 255, 255), 2)
        cv2.line(resized_video, (fx2, fy1), (fx2 - c_len, fy1), (255, 255, 255), 2)
        cv2.line(resized_video, (fx2, fy1), (fx2, fy1 + c_len), (255, 255, 255), 2)
        cv2.line(resized_video, (fx1, fy2), (fx1 + c_len, fy2), (255, 255, 255), 2)
        cv2.line(resized_video, (fx1, fy2), (fx1, fy2 - c_len), (255, 255, 255), 2)
        cv2.line(resized_video, (fx2, fy2), (fx2 - c_len, fy2), (255, 255, 255), 2)
        cv2.line(resized_video, (fx2, fy2), (fx2, fy2 - c_len), (255, 255, 255), 2)

        # Tag badge above box
        label = f"ID #{face.get('track_id', 1)} · {int(face.get('conf', 0.9)*100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = max(0, fy1 - th - 8)
        cv2.rectangle(resized_video, (fx1, tag_y1), (fx1 + tw + 10, fy1), (29, 19, 11), -1)
        cv2.rectangle(resized_video, (fx1, tag_y1), (fx1 + tw + 10, fy1), color, 1)
        cv2.putText(resized_video, label, (fx1 + 5, fy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Landmarks
        for (lx, ly) in face.get('landmarks', []):
            clx, cly = int(lx * sx), int(ly * sy)
            cv2.circle(resized_video, (clx, cly), 2, (187, 244, 123), -1)

    canvas[vy1:vy1+vh, vx1:vx1+vw] = resized_video

    # 3. PLAYBACK BAR (Below video: y: 775 to 845)
    py1 = 780
    cv2.rectangle(canvas, (vx1, py1), (vx1 + vw, py1 + 65), (25, 17, 10), -1)
    cv2.rectangle(canvas, (vx1, py1), (vx1 + vw, py1 + 65), (55, 42, 28), 1)

    # Convert to PIL for crisp text & icons
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Topbar text
    draw.text((25, 18), "VECTOR  /  TRAFFIC & BIOMETRIC INTELLIGENCE", font=fonts['title'], fill=(232, 240, 246))
    draw.text((490, 21), "NETWORK OPERATIONS / LIVE VIEW · PEOPLE MONITOR", font=fonts['small'], fill=(130, 149, 168))

    # Topbar right badges
    draw_rounded_rect(draw, (1420, 15, 1530, 41), 4, fill=(24, 56, 57), outline=(54, 217, 208))
    draw.text((1432, 20), "● LIVE STREAM", font=fonts['badge'], fill=(54, 217, 208))

    draw_rounded_rect(draw, (1545, 15, 1690, 41), 4, fill=(21, 34, 46), outline=(44, 60, 74))
    draw.text((1557, 20), "SOURCE: 4K 60FPS", font=fonts['badge'], fill=(180, 210, 230))

    draw_rounded_rect(draw, (1705, 15, 1885, 41), 4, fill=(21, 34, 46), outline=(44, 60, 74))
    draw.text((1717, 20), "AI: YUNET + RES10 SSD", font=fonts['badge'], fill=(131, 228, 170))

    # Left panel header
    draw.ellipse((38, 84, 46, 92), fill=(54, 217, 208))
    draw.text((56, 78), "RC-264", font=fonts['body_b'], fill=(232, 240, 246))
    draw.text((115, 78), "/ Dubai Walking Tour [4K/60fps]", font=fonts['body'], fill=(130, 149, 168))
    draw_rounded_rect(draw, (1000, 76, 1165, 100), 4, fill=(21, 34, 46), outline=(44, 60, 74))
    draw.text((1012, 80), "YOLO26n · YuNet · ByteTrack", font=fonts['badge'], fill=(130, 149, 168))

    # Right panel header
    draw.text((1220, 78), "Anonymous Person Profiles", font=fonts['title'], fill=(232, 240, 246))
    prof_count_str = f"{len(profiles_data)}"
    draw_rounded_rect(draw, (1475, 78, 1515, 98), 4, fill=(24, 56, 57), outline=(54, 217, 208))
    draw.text((1485, 81), prof_count_str, font=fonts['badge'], fill=(54, 217, 208))
    draw.text((1220, 102), "Verified faces · native source pixels · dual-model confirmed", font=fonts['small'], fill=(130, 149, 168))
    draw_rounded_rect(draw, (1815, 78, 1880, 98), 4, fill=(23, 59, 46), outline=(45, 94, 81))
    draw.text((1825, 81), "SYNCED", font=fonts['badge'], fill=(131, 228, 170))

    # Playback controls in HUD bar
    # Buttons: -30s, Pause, Resume (active), +30s
    bx = vx1 + 16
    draw_rounded_rect(draw, (bx, py1 + 14, bx + 52, py1 + 50), 5, fill=(21, 34, 46), outline=(44, 60, 74))
    draw.text((bx + 11, py1 + 24), "-30s", font=fonts['badge'], fill=(190, 208, 223))

    bx += 60
    draw_rounded_rect(draw, (bx, py1 + 14, bx + 65, py1 + 50), 5, fill=(21, 34, 46), outline=(44, 60, 74))
    draw.text((bx + 12, py1 + 24), "Ⅱ Pause", font=fonts['badge'], fill=(190, 208, 223))

    bx += 73
    draw_rounded_rect(draw, (bx, py1 + 14, bx + 75, py1 + 50), 5, fill=(52, 211, 203), outline=(54, 217, 208))
    draw.text((bx + 10, py1 + 24), "▶ Resume", font=fonts['badge'], fill=(6, 43, 42))

    bx += 83
    draw_rounded_rect(draw, (bx, py1 + 14, bx + 52, py1 + 50), 5, fill=(21, 34, 46), outline=(44, 60, 74))
    draw.text((bx + 11, py1 + 24), "+30s", font=fonts['badge'], fill=(190, 208, 223))

    # Time Readout & Progress Slider
    # current_sec: e.g. 430s = 07:10
    cur_m = int(current_sec // 60)
    cur_s = int(current_sec % 60)
    time_str = f"{cur_m:02d}:{cur_s:02d}"
    draw.text((bx + 75, py1 + 23), time_str, font=fonts['mono'], fill=(54, 217, 208))

    # Timeline scrubber track
    slider_x1 = bx + 140
    slider_x2 = vx1 + vw - 100
    slider_y = py1 + 32
    draw.line([slider_x1, slider_y, slider_x2, slider_y], fill=(44, 60, 74), width=4)

    # Progress fill (relative to total duration or window)
    prog_pct = (current_sec - 430.0) / 47.0 # 0.0 at 7:10, 1.0 at 7:57
    prog_pct = max(0.0, min(1.0, prog_pct))
    slider_fill_x = slider_x1 + int((slider_x2 - slider_x1) * prog_pct)
    if slider_fill_x > slider_x1:
        draw.line([slider_x1, slider_y, slider_fill_x, slider_y], fill=(54, 217, 208), width=4)
        draw.ellipse((slider_fill_x - 6, slider_y - 6, slider_fill_x + 6, slider_y + 6), fill=(255, 255, 255), outline=(54, 217, 208), width=2)

    draw.text((slider_x2 + 15, py1 + 23), "03:35:35", font=fonts['mono'], fill=(130, 149, 168))

    # Video Panel Footer
    draw.text((vx1 + 5, 860), "NATIVE SOURCE  ·  1080P PREVIEW  ·  4K RESOLUTION AI CAPTURES  ·  DUAL-MODEL VERIFIED", font=fonts['small'], fill=(130, 149, 168))

    # Metric strip inside left panel (y: 890 to 990)
    my = 890
    m_w = (vw - 36) // 4
    metrics_data = [
        ("VISIBLE FACES", str(len(active_faces)), "Confirmed face tracks"),
        ("PROFILES CAPTURED", str(len(profiles_data)), "Session gallery verified"),
        ("ENHANCEMENTS", str(len(profiles_data)), "ESPCN-x4 super-resolved"),
        ("AI LATENCY", "17.4 ms", "60 FPS real-time pipeline")
    ]
    for i, (m_label, m_val, m_sub) in enumerate(metrics_data):
        mx = vx1 + i * (m_w + 12)
        draw_rounded_rect(draw, (mx, my, mx + m_w, my + 95), 6, fill=(21, 34, 46), outline=(38, 54, 68))
        draw.text((mx + 14, my + 10), m_label, font=fonts['badge'], fill=(151, 171, 186))
        draw.text((mx + 14, my + 30), m_val, font=fonts['title'], fill=(232, 240, 246))
        draw.text((mx + 14, my + 68), m_sub, font=fonts['small'], fill=(97, 121, 140))

    # 4. RIGHT PANEL: ANONYMOUS PERSON PROFILES CARDS
    # Draw top toolbar
    ry = 135
    draw_rounded_rect(draw, (right_x1 + 16, ry, right_x1 + 350, ry + 36), 5, fill=(15, 26, 36), outline=(38, 54, 68))
    draw.text((right_x1 + 28, ry + 10), "🔍 Search by Track # or Session...", font=fonts['small'], fill=(110, 130, 150))
    draw_rounded_rect(draw, (right_x1 + 365, ry, right_x1 + 510, ry + 36), 5, fill=(15, 26, 36), outline=(38, 54, 68))
    draw.text((right_x1 + 375, ry + 10), "Status: All Profiles ▾", font=fonts['small'], fill=(180, 200, 220))
    draw_rounded_rect(draw, (right_x1 + 525, ry, right_x1 + 680, ry + 36), 5, fill=(15, 26, 36), outline=(38, 54, 68))
    draw.text((right_x1 + 535, ry + 10), "Sort: Newest ▾", font=fonts['small'], fill=(180, 200, 220))

    # Draw profile cards (stack of 3 cards visible simultaneously, scrolling as new ones enter)
    card_y = ry + 52
    card_w = right_x2 - right_x1 - 32 # ~670px
    card_h = 240

    for idx, p in enumerate(profiles_data[:3]): # Top 3 cards
        cy = card_y + idx * (card_h + 16)
        if cy + card_h > right_y2 - 10: break

        # Card container
        draw_rounded_rect(draw, (right_x1 + 16, cy, right_x1 + 16 + card_w, cy + card_h), 10, fill=(16, 28, 40), outline=(44, 66, 88), width=1)
        # Card header bar
        draw_rounded_rect(draw, (right_x1 + 16, cy, right_x1 + 16 + card_w, cy + 38), 10, fill=(22, 36, 50))
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Card Title
        track_id = p.get('track_id', idx + 1)
        draw.text((right_x1 + 32, cy + 10), f"Profile #{track_id}", font=fonts['body_b'], fill=(232, 240, 246))

        # Status badge
        draw_rounded_rect(draw, (right_x1 + 125, cy + 8, right_x1 + 195, cy + 28), 4, fill=(23, 59, 46), outline=(45, 94, 81))
        draw.text((right_x1 + 137, cy + 12), "● ACTIVE", font=fonts['badge'], fill=(131, 228, 170))

        # Quality badge
        q_score = p.get('quality', 88.5)
        draw_rounded_rect(draw, (right_x1 + 205, cy + 8, right_x1 + 300, cy + 28), 4, fill=(24, 56, 57), outline=(54, 217, 208))
        draw.text((right_x1 + 215, cy + 12), f"Score: Q: {q_score:.1f} ★", font=fonts['badge'], fill=(54, 217, 208))

        # Duration
        dur_str = f"Duration: {p.get('duration', 2.4):.1f}s"
        draw.text((right_x1 + 315, cy + 12), dur_str, font=fonts['small'], fill=(130, 149, 168))

        # Paste face crops (Original and Enhanced side-by-side)
        orig_img = p.get('orig_crop')
        enh_img = p.get('enh_crop')

        # Convert back to cv2 for pasting image crops
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        crop_y = cy + 48
        crop_size = 145

        if orig_img is not None:
            ox = right_x1 + 45
            ro = cv2.resize(orig_img, (crop_size, crop_size))
            cv2_img[crop_y:crop_y+crop_size, ox:ox+crop_size] = ro
            cv2.rectangle(cv2_img, (ox, crop_y), (ox+crop_size, crop_y+crop_size), (62, 48, 34), 1)

        if enh_img is not None:
            ex = right_x1 + 225
            re = cv2.resize(enh_img, (crop_size, crop_size))
            cv2_img[crop_y:crop_y+crop_size, ex:ex+crop_size] = re
            cv2.rectangle(cv2_img, (ex, crop_y), (ex+crop_size, crop_y+crop_size), (208, 217, 54), 1)

        # Convert back to PIL
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Tags under crops
        draw_rounded_rect(draw, (right_x1 + 45, crop_y + 4, right_x1 + 105, crop_y + 22), 3, fill=(15, 24, 34), outline=(44, 60, 74))
        draw.text((right_x1 + 52, crop_y + 6), "ORIGINAL", font=fonts['badge'], fill=(180, 200, 220))

        draw_rounded_rect(draw, (right_x1 + 225, crop_y + 4, right_x1 + 295, crop_y + 22), 3, fill=(24, 56, 57), outline=(54, 217, 208))
        draw.text((right_x1 + 232, crop_y + 6), "ENHANCED", font=fonts['badge'], fill=(54, 217, 208))

        # Details on right side of card
        dx = right_x1 + 395
        draw.text((dx, crop_y + 10), "BIOMETRIC ANALYSIS", font=fonts['badge'], fill=(151, 171, 186))
        draw.text((dx, crop_y + 30), f"Resolution: {p.get('res', '128 × 102 px')}", font=fonts['small'], fill=(220, 235, 245))
        draw.text((dx, crop_y + 50), f"Enhancement: {p.get('method', 'ESPCN-x4 + CLAHE')}", font=fonts['small'], fill=(54, 217, 208))
        draw.text((dx, crop_y + 70), f"Verification: {p.get('verif', '99.9% YuNet+Res10')}", font=fonts['small'], fill=(131, 228, 170))
        draw.text((dx, crop_y + 90), f"Pose: {p.get('pose', 'Frontal (yaw +1.4°)')}", font=fonts['small'], fill=(180, 200, 215))
        draw.text((dx, crop_y + 110), f"Session: session_1793690646", font=fonts['badge'], fill=(97, 121, 140))

    # Bottom status bar (y: 1045 to 1080)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(cv2_img, (0, 1045), (1920, 1080), (22, 14, 8), -1)
    cv2.line(cv2_img, (0, 1045), (1920, 1045), (62, 48, 34), 1)
    cv2.putText(cv2_img, "VECTOR / STREET INTELLIGENCE  ·  DUBAI WALKING TOUR 4K STREAM  ·  RECORDING: 07:10 TO 07:57  ·  HIGH-QUALITY FACE CAPTURE",
                (30, 1066), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 170, 190), 1, cv2.LINE_AA)

    return cv2_img

if __name__ == '__main__':
    fonts = get_fonts()
    # Read first frame from test clip
    cap = cv2.VideoCapture('outputs/test_clip_7m10s_7m57s.mp4')
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read test clip"

    dummy_faces = [
        {'box': (750, 320, 110, 140), 'track_id': 101, 'conf': 0.94, 'landmarks': [(780, 360), (830, 360), (805, 395), (785, 425), (825, 425)]},
        {'box': (950, 310, 100, 130), 'track_id': 102, 'conf': 0.91, 'landmarks': [(980, 350), (1020, 350), (1000, 385), (985, 415), (1015, 415)]}
    ]

    # Generate dummy crops for preview
    orig_crop = frame[320:460, 750:860]
    enh_crop = cv2.resize(orig_crop, (orig_crop.shape[1]*2, orig_crop.shape[0]*2), interpolation=cv2.INTER_CUBIC)

    dummy_profiles = [
        {'track_id': 101, 'quality': 92.4, 'duration': 4.8, 'orig_crop': orig_crop, 'enh_crop': enh_crop, 'res': '140 × 110 px', 'method': 'ESPCN-x4 + CLAHE', 'verif': '99.9% YuNet+Res10', 'pose': 'Frontal (yaw -0.8°)'},
        {'track_id': 102, 'quality': 89.1, 'duration': 3.2, 'orig_crop': orig_crop, 'enh_crop': enh_crop, 'res': '130 × 100 px', 'method': 'ESPCN-x4 + CLAHE', 'verif': '99.8% YuNet+Res10', 'pose': 'Frontal (yaw +2.1°)'},
        {'track_id': 98, 'quality': 86.7, 'duration': 6.5, 'orig_crop': orig_crop, 'enh_crop': enh_crop, 'res': '125 × 95 px', 'method': 'ESPCN-x4 + CLAHE', 'verif': '99.7% YuNet+Res10', 'pose': 'Slight Right (yaw +7.4°)'}
    ]

    out_img = render_composite_frame(frame, 430.0, dummy_profiles, dummy_faces, fonts)
    cv2.imwrite('outputs/preview_composite_frame.jpg', out_img)
    print("Rendered preview composite frame to outputs/preview_composite_frame.jpg! Size:", os.path.getsize('outputs/preview_composite_frame.jpg'))
