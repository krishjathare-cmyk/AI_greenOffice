# ai_script.py
from google import genai
from google.genai import types
from PIL import Image, ImageChops, ImageOps
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY").strip('"\' ')
client = genai.Client(api_key=api_key)
GLASSES_MODEL = os.getenv("GEMINI_GLASSES_MODEL", "gemini-3.5-flash").strip('"\' ')
RX_MODEL = os.getenv("GEMINI_RX_MODEL", GLASSES_MODEL).strip('"\' ')
RX_MAX_DIMENSION = int(os.getenv("GEMINI_RX_MAX_DIMENSION", "1000"))
RX_CROP_BORDER = int(os.getenv("GEMINI_RX_CROP_BORDER", "24"))


def analyze_glasses_image(image_path: str):
    img = Image.open(image_path)

    response = client.models.generate_content(
        model=GLASSES_MODEL,
        contents=[
            """
            Analyze the eyeglasses in this image for an inventory form.

            Return ONLY valid JSON. Do not include markdown, comments, or extra text.
            Use exactly the enum/string values listed below so the result can fill a web form.

            Allowed values:
            - frame_material: "METAL", "ACETATE", "PLASTIC", "TITANIUM", "STAINLESS_STEEL", "MIXED", "OTHER", or null
            - frame_color: array using one or more of "BLACK", "GREY", "WHITE", "GOLD", "SILVER", "BRONZE", "BROWN", "TORTOISE", "TRANSPARENT", "BLUE", "RED", "GREEN", "YELLOW", "ORANGE", "PURPLE", "PINK", "BEIGE", "MULTICOLOR"
            - intended_for: "UNISEX", "WOMEN", "MEN", "KIDS", or null
            - fit_size: "S", "M", "L", or null
            - lens_design: "SINGLE_VISION", "PROGRESSIVE", "OCCUPATIONAL", "BIFOCAL", "TRIFOCAL", or null
            - lens_material: "PLASTIC", "POLYCARBONATE", "TRIVEX", "GLASS", "OTHER", or null
            - item_condition: "EXCELLENT", "GOOD", "FAIR", or null

            Rules:
            - Choose the best visible value. Use null only when the image gives no useful clue.
            - Frame material:
              - thin shiny wire frames usually mean "METAL".
              - thick molded frames usually mean "ACETATE" or "PLASTIC".
              - mixed metal plus plastic/acetate parts means "MIXED".
              - use "TITANIUM" or "STAINLESS_STEEL" only if it is clearly identifiable; otherwise use "METAL".
            - Frame color:
              - return all important visible frame colors, most dominant first.
              - tortoise/havana patterns must be "TORTOISE".
              - clear/crystal frames must be "TRANSPARENT".
              - if there are many colors or a pattern not covered, use "MULTICOLOR".
            - intended_for:
              - use "KIDS" only for clearly small/children frames.
              - use "WOMEN" or "MEN" only when style strongly suggests it.
              - otherwise use "UNISEX".
            - fit_size:
              - estimate from apparent frame width: narrow/small = "S", average = "M", wide/oversized = "L".
            - lens_design:
              - if no visible line or special segment exists, usually use "SINGLE_VISION".
              - visible bifocal line/segment means "BIFOCAL"; three zones means "TRIFOCAL".
              - progressive lenses usually have no visible line, so only choose "PROGRESSIVE" if there is clear evidence.
            - lens_material:
              - most modern eyeglass lenses should be "PLASTIC" unless clearly glass or another material.
            - item_condition:
              - "EXCELLENT" means the glasses look almost new, clean, with no visible scratches, bent parts, or damage.
              - "GOOD" means usable with minor wear, small cosmetic marks, or normal second-hand condition.
              - "FAIR" means visibly worn but still usable, with noticeable scratches, bent frame, missing nose pad, loose temple, or other visible defects.
              - judge only from visible image evidence; if unsure, use "GOOD".
            - features:
              - anti_reflective: true if lenses show strong anti-reflective tint/coating, usually green/blue/purple reflections.
              - blue_light_filter: true if lenses show a clear blue-light filtering tint/coating.
              - anti_fog: true only if clearly stated/visible; usually false from image alone.
              - clip_on: true if there is a clip-on sunglass attachment or magnetic clip visible.
              - photochromic: true only if transition/photochromic tint is clearly visible.
              - sunglasses: true if lenses are visibly dark/tinted like sunglasses.


            {
              "frame_material": null,
              "frame_color": [],
              "intended_for": null,
              "fit_size": null,
              "lens_design": null,
              "lens_material": null,
              "item_condition": null,
              "features": {
                "anti_reflective": false,
                "blue_light_filter": false,
                "anti_fog": false,
                "clip_on": false,
                "photochromic": false,
                "sunglasses": false
              }
            }
            """,
            img,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    return json.loads(response.text)


def _normalize_rx_number(value):
    if value is None:
        return None

    cleaned = str(value).strip().replace(",", ".")
    cleaned = cleaned.replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"^\+\+", "+", cleaned)
    cleaned = re.sub(r"^--", "-", cleaned)
    cleaned = re.sub(r"^([+-])\.", r"\g<1>0.", cleaned)

    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_eye_values(text, marker_pattern):
    match = re.search(marker_pattern, text, flags=re.IGNORECASE)
    if not match:
        return {"sph": None, "cyl": None, "axis": None, "add": None}

    segment = text[match.end():]
    next_marker = re.search(r"\b(R|RIGHT|RECHTS|OD|L|LEFT|LINKS|OS)\b", segment, flags=re.IGNORECASE)
    if next_marker:
        segment = segment[:next_marker.start()]

    numbers = re.findall(r"[+-]?\s*(?:\d+(?:[.,]\d+)?|[.,]\d+)", segment)
    values = [_normalize_rx_number(number.replace(" ", "")) for number in numbers]
    values = [value for value in values if value is not None]

    axis = None
    if len(values) >= 3:
        axis_candidate = int(round(values[2]))
        axis = axis_candidate if 0 <= axis_candidate <= 180 else None

    return {
        "sph": values[0] if len(values) >= 1 else None,
        "cyl": values[1] if len(values) >= 2 else None,
        "axis": axis,
        "add": values[3] if len(values) >= 4 else None,
    }


def _parse_prescription_text(text):
    normalized = text.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized)

    right = _parse_eye_values(normalized, r"\b(R|RIGHT|RECHTS|OD)\b")
    left = _parse_eye_values(normalized, r"\b(L|LEFT|LINKS|OS)\b")

    return {
        "right": right,
        "left": left,
        "pd": None,
        "raw_text": text,
    }


def _prepare_prescription_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    cropped = _crop_prescription_region(img)
    resized = _resize_for_rx(cropped)
    return resized


def _crop_prescription_region(img: Image.Image):
    grayscale = ImageOps.grayscale(img)
    boosted = ImageOps.autocontrast(grayscale)
    background = Image.new("L", boosted.size, 255)
    diff = ImageChops.difference(boosted, background)
    # Ignore tiny noise and keep the main paper/text region when possible.
    mask = diff.point(lambda value: 255 if value < 242 else 0)
    bbox = ImageOps.invert(mask).getbbox()

    if bbox is None:
        return img

    left, top, right, bottom = bbox
    left = max(0, left - RX_CROP_BORDER)
    top = max(0, top - RX_CROP_BORDER)
    right = min(img.width, right + RX_CROP_BORDER)
    bottom = min(img.height, bottom + RX_CROP_BORDER)

    cropped = img.crop((left, top, right, bottom))
    if cropped.width < img.width * 0.35 or cropped.height < img.height * 0.2:
        return img
    return cropped


def _resize_for_rx(img: Image.Image):
    if max(img.size) <= RX_MAX_DIMENSION:
        return img

    width, height = img.size
    if width >= height:
        new_width = RX_MAX_DIMENSION
        new_height = int(round(height * RX_MAX_DIMENSION / width))
    else:
        new_height = RX_MAX_DIMENSION
        new_width = int(round(width * RX_MAX_DIMENSION / height))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def analyze_prescription_image(image_path: str):
    img = _prepare_prescription_image(image_path)

    response = client.models.generate_content(
        model=RX_MODEL,
        contents=[
            """
            Read this eyeglass prescription image and extract only the visible prescription values.

            Return ONLY valid JSON with this exact structure:
            {
              "right": { "sph": null, "cyl": null, "axis": null, "add": null },
              "left": { "sph": null, "cyl": null, "axis": null, "add": null },
              "pd": null
            }

            Rules:
            - Use the right eye for R, RIGHT, RECHTS, or OD.
            - Use the left eye for L, LEFT, LINKS, or OS.
            - Numbers may use either comma or dot decimals.
            - axis must be an integer from 0 to 180, or null.
            - pd should be a number if visible, otherwise null.
            - If a field is not clearly visible, use null.
            - Do not include markdown or extra explanation.
            """,
            img,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    payload = json.loads(response.text)
    result = {
        "right": {"sph": None, "cyl": None, "axis": None, "add": None},
        "left": {"sph": None, "cyl": None, "axis": None, "add": None},
        "pd": None,
        "raw_text": "",
    }

    for eye in ("right", "left"):
        eye_data = payload.get(eye, {}) or {}
        for field in ("sph", "cyl", "add"):
            normalized = _normalize_rx_number(eye_data.get(field))
            if normalized is not None:
                result[eye][field] = normalized
        axis_value = eye_data.get("axis")
        axis_normalized = _normalize_rx_number(axis_value)
        if axis_normalized is not None:
            axis_candidate = int(round(axis_normalized))
            if 0 <= axis_candidate <= 180:
                result[eye]["axis"] = axis_candidate

    pd_normalized = _normalize_rx_number(payload.get("pd"))
    result["pd"] = pd_normalized if pd_normalized is not None else result["pd"]

    return result
