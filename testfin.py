"""
Adaptive document deskewing, following Bao et al. 2022 (Sensors 22, 7944).

Pipeline:
  1. Image Classification (IC)      -> text / form / complex
  2. Text     -> Skeleton Line Detection (SKLD) + Piecewise Projection Profile (PPP)
  3. Form     -> Hough line detection + outlier elimination
  4. Complex  -> Morphological Clustering (MC) + Fourier transform

Usage:
    python deskew.py input.jpg output.jpg
    from deskew import deskew_image
    angle, corrected, img_type = deskew_image(img_bgr)
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def _binarize(gray):
    """Adaptive binarization -> foreground=255 (text/ink), background=0."""
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return bin_img


def _rotate(img, angle, border_value=255, expand_canvas=False):
    """
    Rotate img (grayscale/BGR) about its center by angle degrees.
    If expand_canvas=False (default), keeps the same canvas dimensions matching rotation.py.
    """
    if abs(angle) < 1e-4:
        return img.copy()
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if expand_canvas:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(np.ceil(h * sin + w * cos))
        new_h = int(np.ceil(h * cos + w * sin))
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0
    else:
        new_w, new_h = w, h

    return cv2.warpAffine(
        img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
    )


# ----------------------------------------------------------------------
# 3.2  Image Classification
#      Eq.(1): contour aspect-ratio test  -> text vs non-text
#      Eq.(2): line count / slope-variance test -> form vs complex
# ----------------------------------------------------------------------
def classify_image(bin_img, gray=None):
    """
    Classify document for deskewing.

    Priority:
        1. FORM  -> strong horizontal/vertical structure (tables/forms)
        2. TEXT  -> mostly text-like structure
        3. COMPLEX -> everything else

    For invoice/table datasets, FORM is intentionally detected first
    because table lines are the most useful structures for Hough deskewing.
    """

    h, w = bin_img.shape

    # ---------------------------------------------------------
    # 1. Detect horizontal and vertical document structures
    # ---------------------------------------------------------

    # Horizontal structures
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(10, w // 30), 1)
    )

    horizontal = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # Vertical structures
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(10, h // 30))
    )

    vertical = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    # ---------------------------------------------------------
    # 2. Measure amount of horizontal / vertical structure
    # ---------------------------------------------------------

    horizontal_pixels = cv2.countNonZero(horizontal)
    vertical_pixels = cv2.countNonZero(vertical)

    image_area = h * w

    horizontal_ratio = horizontal_pixels / image_area
    vertical_ratio = vertical_pixels / image_area

    # ---------------------------------------------------------
    # 3. Count actual long lines
    # ---------------------------------------------------------

    contours_h, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours_v, _ = cv2.findContours(
        vertical,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    long_horizontal = 0
    long_vertical = 0

    min_horizontal_length = w * 0.15
    min_vertical_length = h * 0.15

    for c in contours_h:
        x, y, cw, ch = cv2.boundingRect(c)

        if cw >= min_horizontal_length:
            long_horizontal += 1

    for c in contours_v:
        x, y, cw, ch = cv2.boundingRect(c)

        if ch >= min_vertical_length:
            long_vertical += 1

    # ---------------------------------------------------------
    # 4. FORM / TABLE detection
    # ---------------------------------------------------------

    # A document with several long horizontal/vertical structures
    # is treated as a form/table document.

    if (
        long_horizontal >= 2
        or long_vertical >= 2
        or (
            horizontal_ratio > 0.001
            and vertical_ratio > 0.0005
        )
    ):
        return "form"

    # ---------------------------------------------------------
    # 5. TEXT detection
    # ---------------------------------------------------------

    # If there are no strong table/form structures,
    # inspect connected components.

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_img,
        connectivity=8
    )

    text_like_components = 0

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        # Typical text-like component
        if (
            5 <= cw <= w * 0.5
            and 3 <= ch <= h * 0.1
            and area >= 10
        ):
            text_like_components += 1

    # Lots of small text components and no strong table structure
    # → text document.
    if text_like_components >= 20:
        return "text"

    # ---------------------------------------------------------
    # 6. Otherwise
    # ---------------------------------------------------------

    return "complex"

# ----------------------------------------------------------------------
# 3.3  Text image correction: SKLD + PPP
# ----------------------------------------------------------------------

def _writing_direction(bin_img):
    """LSD line detection -> horizontal vs vertical writing direction."""
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(bin_img)[0]
    if lines is None:
        return "horizontal"
    Lh = Lv = 0
    for l in lines:
        x1, y1, x2, y2 = np.asarray(l).ravel()
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) > 1:
            Lv += 1
        else:
            Lh += 1
    return "horizontal" if Lh >= Lv else "vertical"


def _skeleton_line_angle(bin_img, direction):
    """
    SKLD: expand text blobs (M<N or M>N depending on direction), keep
    elongated contours, draw their min-area rectangles, thin (Zhang-Suen)
    to a skeleton, and estimate the dominant line angle via Hough on the
    skeleton.
    """
    h, w = bin_img.shape
    if direction == "vertical":
        M, N = max(15, h // 40), max(3, w // 100)   # M > N
    else:
        M, N = max(3, h // 100), max(15, w // 40)   # M < N
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (N, M))
    dilated = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rect_img = np.zeros_like(bin_img)
    x_ratio = 2.0  # minimum elongation to keep a contour as a "text line"
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw == 0 or ch == 0:
            continue
        ar = (cw / ch) if direction == "horizontal" else (ch / cw)
        if ar >= x_ratio:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(rect_img, [box], -1, 255, -1)

    if not np.any(rect_img):
        return 0.0

    skeleton = cv2.ximgproc.thinning(rect_img)

    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(skeleton)[0]
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for l in lines:
        x1, y1, x2, y2 = np.asarray(l).ravel()
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 15:
            continue
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # normalize into (-45, 45] relative to writing direction
        if direction == "horizontal":
            if ang > 90:
                ang -= 180
            elif ang < -90:
                ang += 180
            if ang > 45:
                ang -= 90
            elif ang < -45:
                ang += 90
        else:
            ang = ang - 90 if ang > 0 else ang + 90
        angles.append(ang)

    if not angles:
        return 0.0
    return float(np.median(angles))


def _valley_value(profile):
    """Eq.(4): sum of foreground pixel counts in columns 5..9 of the profile
    (the low, dense region near the projection base)."""
    # For a projection value v, the number of pixels in column indices 5..9 is:
    # 0 if v <= 5, v - 5 if 5 < v < 10, and 5 if v >= 10.
    # This is equivalent to clipping (profile - 5) to [0, 5].
    contributions = np.clip(profile - 5, 0, 5)
    return float(np.sum(contributions))


def _projection_profile(bin_img, angle, direction):
    rotated = _rotate(bin_img, angle, border_value=0)
    axis = 1 if direction == "horizontal" else 0
    profile = np.sum(rotated > 0, axis=axis).astype(np.float64)
    return profile


def piecewise_projection_profile(bin_img, direction, l1=0.5, l2=0.05,
                                  theta_start=-5.0, theta_end=5.0):
    """
    PPP (Sec 3.3 / Algorithm 1), two-stage coarse-to-fine search that
    minimizes the valley value (Eq. 3-6).
    """
    def scan(start, end, step):
        angles = np.arange(start, end + step / 2, step)
        vals = []
        for a in angles:
            prof = _projection_profile(bin_img, a, direction)
            vals.append(_valley_value(prof))
        vals = np.array(vals)
        min_val = vals.min()
        best = angles[np.isclose(vals, min_val, rtol=1e-6, atol=1e-6)]
        return best

    # Stage 1: coarse search
    best1 = scan(theta_start, theta_end, l1)
    if len(best1) > 1:
        new_start, new_end = best1.min(), best1.max()
    else:
        new_start, new_end = best1[0] - l1, best1[0] + l1

    # Stage 2: fine search
    best2 = scan(new_start, new_end, l2)
    return float(np.mean(best2))


def correct_text_image(gray, bin_img):
    direction = _writing_direction(bin_img)

    # Step 1: SKLD coarse angle
    try:
        skld_angle = _skeleton_line_angle(bin_img, direction)
    except Exception:
        skld_angle = 0.0
    skld_angle = float(np.clip(skld_angle, -45, 45))

    pre_rotated_bin = _rotate(bin_img, skld_angle, border_value=0)

    # Step 2: PPP refinement, searched around 0 after SKLD pre-correction
    ppp_angle = piecewise_projection_profile(
        pre_rotated_bin, direction, l1=0.1, l2=0.01,
        theta_start=-2.0, theta_end=2.0
    )

    total_angle = skld_angle + ppp_angle
    corrected = _rotate(gray, total_angle, border_value=255)
    return total_angle, corrected


# ----------------------------------------------------------------------
# 3.4  Form image correction: Hough line detection + outlier elimination
# ----------------------------------------------------------------------

def correct_form_image(gray, bin_img, outlier_window=0.5):
    edges = cv2.Canny(bin_img, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=5
    )
    if lines is None:
        return 0.0, gray.copy()

    angles = []
    for l in lines:
        x1, y1, x2, y2 = np.asarray(l).ravel()
        if x2 == x1:
            theta = 90.0
        else:
            k = (y2 - y1) / (x2 - x1)
            theta = np.degrees(np.arctan(k))
        # Eq.(9): normalize to [-45, 45]
        if theta < -45:
            theta += 90
        elif theta > 45:
            theta -= 90
        angles.append(theta)
    angles = np.array(angles)

    # Eq.(10): mean, then Eq.(9-region) outlier elimination within +/- a
    mean_angle = float(np.mean(angles))
    a = outlier_window
    filtered = angles[np.abs(angles - mean_angle) <= a]
    if len(filtered) == 0:
        filtered = angles

    final_angle = float(np.mean(filtered))
    corrected = _rotate(gray, final_angle, border_value=255)
    return final_angle, corrected


# ----------------------------------------------------------------------
# 3.5  Complex content image correction: Morphological Clustering + FFT
# ----------------------------------------------------------------------

def correct_complex_image(gray, bin_img, area_thresh=100):
    h, w = bin_img.shape
    M = max(5, h // 60)
    N = max(5, w // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (N, M))
    clustered = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clustered, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Ic = np.zeros_like(bin_img)
    for c in contours:
        if cv2.contourArea(c) >= area_thresh:
            cv2.drawContours(Ic, [c], -1, 255, 1)

    if not np.any(Ic):
        return 0.0, gray.copy()

    # Fourier transform -> magnitude spectrum
    f = np.fft.fft2(Ic.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # Suppress the DC spike at the center so it doesn't dominate thresholding.
    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    r = max(3, min(magnitude.shape) // 40)
    mag_masked = magnitude.copy()
    cv2.circle(mag_masked, (cx, cy), r, 0, -1)

    # Keep only the brightest ridge (the high-frequency axis lines visible
    # in Fig. 6) via a high percentile threshold -- far more selective than
    # Otsu on a spectrum dominated by low-level speckle.
    thresh_val = np.percentile(mag_masked, 99.0)
    mag_bin = (mag_masked >= thresh_val).astype(np.uint8) * 255
    mag_bin = cv2.dilate(mag_bin, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(
        mag_bin, 1, np.pi / 360, threshold=25, minLineLength=25, maxLineGap=4
    )
    if lines is None:
        return 0.0, gray.copy()

    angles = []
    for l in lines:
        x1, y1, x2, y2 = np.asarray(l).ravel()
        if x2 == x1:
            continue
        theta = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
        if theta < -45:
            theta += 90
        elif theta > 45:
            theta -= 90
        angles.append(theta)

    if not angles:
        return 0.0, gray.copy()

    final_angle = float(np.mean(angles))
    corrected = _rotate(gray, final_angle, border_value=255)
    return final_angle, corrected


# ----------------------------------------------------------------------
# Bounding Box Extraction & Visualization
# ----------------------------------------------------------------------

def get_layout_bboxes(img_bgr):
    """Detects text blocks/lines as bounding boxes [x, y, w, h] in the BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    bin_img = _binarize(gray)
    h, w = bin_img.shape
    
    # Run morphological closing to group letters/words into lines
    M = max(3, h // 120)
    N = max(10, w // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (N, M))
    dilated = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 8 and ch > 8 and cw < w * 0.98 and ch < h * 0.98:
            bboxes.append([x, y, cw, ch])
            
    # Sort bboxes top-to-bottom, left-to-right
    bboxes.sort(key=lambda b: (b[1], b[0]))
    return bboxes


def rotate_bboxes(bboxes, angle, orig_shape, new_shape, un_envelope=True):
    """
    Rotate axis-aligned bboxes using the exact same affine transform
    applied to the image in _rotate().
    Accepts either list of dicts [{"label", "x", "y", "w", "h"}] or list of [x, y, w, h].
    When un_envelope=True, analytically reconstructs the true straightened width/height
    from the rotated axis-aligned envelope to avoid multi-rotation dilation.
    Preserves 1:1 list order without sorting.
    """
    if abs(angle) < 1e-4 or len(bboxes) == 0:
        return [b.copy() if isinstance(b, dict) else list(b) for b in bboxes]

    h, w = orig_shape[:2]
    new_h, new_w = new_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # Adjust translation to account for enlarged canvas (exact same as _rotate)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rad = np.radians(abs(angle))
    cos_t = np.cos(rad)
    sin_t = np.sin(rad)
    cos_2t = np.cos(2.0 * rad)

    rotated_bboxes = []
    for item in bboxes:
        if isinstance(item, dict):
            x, y, bw, bh = item["x"], item["y"], item["w"], item["h"]
            label = item.get("label", "")
            is_dict = True
        else:
            x, y, bw, bh = item
            label = ""
            is_dict = False

        # Center of the rotated bbox
        bcx = x + bw / 2.0
        bcy = y + bh / 2.0

        # Rotate the center point
        nbcx = M[0, 0] * bcx + M[0, 1] * bcy + M[0, 2]
        nbcy = M[1, 0] * bcx + M[1, 1] * bcy + M[1, 2]

        # Analytically recover the original unrotated width and height
        if un_envelope and cos_2t > 0.05:
            rec_w = (bw * cos_t - bh * sin_t) / cos_2t
            rec_h = (bh * cos_t - bw * sin_t) / cos_2t
            rec_w = float(bw) if rec_w <= 1.0 else rec_w
            rec_h = float(bh) if rec_h <= 1.0 else rec_h
        else:
            rec_w, rec_h = float(bw), float(bh)

        rx_min = int(round(nbcx - rec_w / 2.0))
        ry_min = int(round(nbcy - rec_h / 2.0))
        rx_max = int(round(nbcx + rec_w / 2.0))
        ry_max = int(round(nbcy + rec_h / 2.0))

        # Clip to new image bounds
        rx_min = max(0, min(new_w, rx_min))
        ry_min = max(0, min(new_h, ry_min))
        rx_max = max(0, min(new_w, rx_max))
        ry_max = max(0, min(new_h, ry_max))

        rbw = rx_max - rx_min
        rbh = ry_max - ry_min

        if rbw > 0 and rbh > 0:
            if is_dict:
                rotated_bboxes.append({"label": label, "x": rx_min, "y": ry_min, "w": rbw, "h": rbh})
            else:
                rotated_bboxes.append([rx_min, ry_min, rbw, rbh])

    return rotated_bboxes


def visualize_bboxes(img_bgr, bboxes):
    """Draws green bounding boxes (with optional labels) on a copy of the image."""
    viz = img_bgr.copy()
    for item in bboxes:
        if isinstance(item, dict):
            x, y, w, h = item["x"], item["y"], item["w"], item["h"]
            label = item.get("label", "")
        else:
            x, y, w, h = item
            label = ""
        cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if label:
            cv2.putText(viz, label, (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)
    return viz


# ----------------------------------------------------------------------
# XML I/O  (Pascal VOC format: xmin/ymin/xmax/ymax)
# ----------------------------------------------------------------------

def parse_voc_xml(xml_source):
    """
    Parse a Pascal-VOC XML file/string/bytes.

    Args:
        xml_source: file path (str/Path), raw XML bytes, or XML string.

    Returns:
        list of dicts: [{"label": str, "x": int, "y": int, "w": int, "h": int}, ...]
    """
    if isinstance(xml_source, (bytes, bytearray)):
        root = ET.fromstring(xml_source.decode())
    elif isinstance(xml_source, str) and xml_source.strip().startswith("<"):
        root = ET.fromstring(xml_source)
    else:
        root = ET.parse(xml_source).getroot()

    bboxes = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="")
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = int(float(bb.findtext("xmin", "0")))
        ymin = int(float(bb.findtext("ymin", "0")))
        xmax = int(float(bb.findtext("xmax", "0")))
        ymax = int(float(bb.findtext("ymax", "0")))
        bboxes.append({"label": label,
                        "x": xmin, "y": ymin,
                        "w": xmax - xmin, "h": ymax - ymin})
    return bboxes


def write_voc_xml(bboxes, img_shape, filename="image"):
    """
    Serialize rotated bboxes back to Pascal-VOC XML string.

    Args:
        bboxes: list of dicts [{"label", "x", "y", "w", "h"}, ...]
        img_shape: (height, width[, channels]) of the output image
        filename: value for the <filename> element

    Returns:
        XML string (utf-8)
    """
    h, w = img_shape[:2]
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text  = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text  = str(img_shape[2] if len(img_shape) > 2 else 1)
    for item in bboxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = item.get("label", "")
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(item["x"])
        ET.SubElement(bb, "ymin").text = str(item["y"])
        ET.SubElement(bb, "xmax").text = str(item["x"] + item["w"])
        ET.SubElement(bb, "ymax").text = str(item["y"] + item["h"])
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def deskew_image(img_bgr, return_bboxes=True, xml_bboxes=None):
    """
    Deskew an image and optionally transform its bounding boxes using the exact same affine matrix.

    Args:
        img_bgr:       Input BGR image (numpy array).
        return_bboxes: If True, also return bbox arrays and visualizations.
        xml_bboxes:    Optional list of bbox dicts from parse_voc_xml().
                       If provided, these are rotated instead of auto-detecting.
                       Format: [{"label": str, "x": int, "y": int, "w": int, "h": int}, ...]

    Returns:
        return_bboxes=True:
            (angle, corrected, img_type, bboxes_before, bboxes_after, viz_before, viz_after)
        return_bboxes=False:
            (angle, corrected, img_type)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    bin_img = _binarize(gray)

    # Estimate deskew angle using Hough line detection on tables/forms
    angle, _ = correct_form_image(gray, bin_img)
    img_type = 'form'

    # Single source-of-truth manual warp with expand-canvas logic
    border_val = (255, 255, 255) if img_bgr.ndim == 3 else 255
    corrected = _rotate(img_bgr, angle, border_value=border_val, expand_canvas=True)

    if return_bboxes:
        if xml_bboxes is not None:
            bboxes_before = xml_bboxes
        else:
            # Convert auto-detected [x,y,w,h] lists to dicts for uniform handling
            bboxes_before = [
                {"label": "", "x": b[0], "y": b[1], "w": b[2], "h": b[3]}
                for b in get_layout_bboxes(img_bgr)
            ]

        # Transform bboxes with the exact same M and canvas dimensions
        bboxes_after = rotate_bboxes(
            bboxes_before,
            angle,
            orig_shape=img_bgr.shape,
            new_shape=corrected.shape
        )

        viz_before = visualize_bboxes(img_bgr, bboxes_before)
        viz_after  = visualize_bboxes(corrected, bboxes_after)
        return angle, corrected, img_type, bboxes_before, bboxes_after, viz_before, viz_after

    return angle, corrected, img_type


def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python deskew.py <input_image> <output_image>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise FileNotFoundError(sys.argv[1])

    angle, corrected, img_type = deskew_image(img, return_bboxes=False)
    cv2.imwrite(sys.argv[2], corrected)
    print(f"Type: {img_type} | Estimated skew: {angle:.3f} deg | Saved: {sys.argv[2]}")


if __name__ == "__main__":
    main()
