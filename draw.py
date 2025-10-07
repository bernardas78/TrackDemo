import cv2
import colorsys

def draw_rectangle(img, x1, y1, x2, y2, track_id):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    col = int_to_color(track_id)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), col, thickness=2)


def int_to_color(n: int): #-> tuple[int, int, int]:
    """
    Map an integer to a distinct RGB color.
    Nearby integers produce very different colors.
    Returns a tuple (R, G, B) in 0–255.
    """
    # Use the golden ratio to distribute hues uniformly
    golden_ratio_conjugate = 0.618033988749895
    hue = (n * golden_ratio_conjugate) % 1.0  # gives good separation between nearby ints

    # Use full saturation and brightness for vivid colors
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)

    # Convert to 0–255 range
    return (int(r * 255), int(g * 255), int(b * 255))