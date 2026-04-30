"""
Shared configuration constants for the Demand Signal Pipeline.
Mirrored from the original notebook logic to ensure consistency.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR      = BASE_DIR / "data"
OUTPUT_DIR    = DATA_DIR / "demand_signals"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LINKED_DIR    = DATA_DIR / "linked_data"
CLEANED_DIR   = DATA_DIR / "cleaned_data"

for _dir in [OUTPUT_DIR, CHECKPOINT_DIR, LINKED_DIR, CLEANED_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Keyword generation: object categories ───────────────────────────────────────
OBJECT_CATEGORIES = {
    "photography": [
        "camera", "DSLR", "mirrorless camera", "camera lens",
        "GoPro", "drone",
    ],
    "musical_instruments": [
        "guitar", "violin", "drums", "keyboard piano",
        "synthesizer", "DJ controller",
    ],
    "outdoor_camping": [
        "tent", "sleeping bag", "camping cookware", "fishing gear",
        "climbing gear", "portable stove",
    ],
    "travel_luggage": [
        "luggage", "suitcase", "travel bag", "toiletry bag",
        "cable organizer", "packing cube",
    ],
    "gaming_console": [
        "Steam Deck", "Nintendo Switch", "PlayStation",
        "gaming headset", "VR headset", "gaming mouse",
    ],
    "tools_equipment": [
        "power tools", "drill", "oscilloscope", "soldering iron",
        "3D printer", "multimeter",
    ],
    "collectibles": [
        "figurine", "action figure", "PVC statue", "anime figure",
        "Lego set", "vinyl record",
    ],
    "electronic_accessories": [
        "hard drive", "SSD", "external storage", "USB drive",
        "laptop", "power bank",
    ],
    "eyewear_optics": [
        "sunglasses", "eyeglasses", "goggles", "binoculars",
        "telescope", "spectacles",
    ],
    "kitchen_cookware": [
        "chef knife", "kitchen knife set", "cast iron skillet",
        "wine glass", "espresso machine", "blender",
    ],
    "sports_fitness": [
        "golf clubs", "tennis racket", "bicycle", "skateboard",
        "surfboard", "dumbbells",
    ],
    "medical_health": [
        "CPAP machine", "hearing aid", "blood pressure monitor",
        "glucose meter", "inhaler", "first aid kit",
    ],
    "baby_kids": [
        "stroller", "car seat", "baby monitor", "toys",
        "building blocks", "kids camera",
    ],
    "instrument_accessories": [
        "pedal", "amplifier", "effects pedal", "instrument cable",
        "tuner", "metronome",
    ],
    "drone_rc": [
        "drone", "RC car", "RC plane", "FPV drone",
        "racing drone", "RC helicopter",
    ],
    "beauty_personal": [
        "hair dryer", "straightener", "curling iron",
        "electric toothbrush", "shaver", "skincare device",
    ],
    "art_crafts": [
        "paint set", "art supplies", "sketchbook",
        "crafting tools", "sewing machine", "embroidery kit",
    ],
    "jewelry_watches": [
        "watch", "jewelry", "necklace", "bracelet",
        "ring", "earrings",
    ],
}

PROTECTION_QUERIES = [
    "{obj} hard case",
    "{obj} protective case",
]

STORAGE_QUERIES = [
    "{obj} storage case",
    "{obj} carry case",
]

USAGE_QUERIES = [
    "{obj} everyday carry",
    "{obj} EDC setup",
]

PROBLEM_QUERIES = [
    "{obj} damaged",
    "{obj} best case for",
]

LEADER_DISCOVERY_QUERIES = [
    "hard case review",
    "protective case for valuable items",
    "best carry case",
    "storage solution for fragile items",
    "gear organizer",
    "equipment storage",
    "cables organizer",
    "tech organizer bag",
    "travel fragile items",
    "how to protect fragile gear",
    "packing fragile equipment",
    "fragile items travel case",
    "best protective case",
    "shockproof case review",
    "waterproof case for gear",
    "crush proof case",
    "impact protection case",
    "damage protection case",
    "safe transport case",
    "hard shell case review",
    "protective case comparison",
    "best hard case brand",
    "durable case for expensive equipment",
    "heavy duty protective case",
    "compact storage case",
    "organizer case review",
    "carrying case recommendation",
    "travel case for valuables",
    "portable case solution",
    "fragile equipment storage",
    "professional gear case",
    "storage case for hobby",
    "travel setup for gear",
    "EDC carry case",
    "everyday carry organizer",
    "gear bag setup",
    "kit bag organization",
    "packing solution for equipment",
    "how to pack fragile gear",
    "travel packing tips gear",
    "best packing method fragile",
    "moving fragile items",
    "shipping fragile equipment",
    "best case for camera gear",
    "best case for camera lens",
    "best case for drone",
    "best case for guitar",
    "best case for violin",
    "best case for keyboard",
    "best case for synthesizer",
    "best case for DJ controller",
    "best case for laptop",
    "best case for hard drive",
    "best case for Steam Deck",
    "best case for Nintendo Switch",
    "best case for gaming headset",
    "best case for power tools",
    "best case for drill",
    "best case for oscilloscope",
    "best case for soldering iron",
    "best case for tent",
    "best case for sleeping bag",
    "best case for fishing gear",
    "best case for headphones",
    "best case for figurines",
    "best case for collectibles",
    "best case for vinyl records",
    "best case for art supplies",
    "best case for makeup",
    "best case for tools",
    "best case for fragile items",
    "best protective case for travel",
    "best storage case organization",
    "how to store camera equipment",
    "how to store guitar at home",
    "how to transport drone",
    "how to pack fragile decorations",
    "how to ship collectibles safely",
    "moving fragile belongings",
    "best way to pack fragile items",
    "travel fragile belongings protection",
]

NEGATIVE_TERMS = [
    "iphone case", "phone case", "smartphone case",
    "android phone", "samsung case", "google pixel",
    "ipad case", "tablet case",
    "fashion case", "cute case", "aesthetic case",
    "decorative case", "phone skin",
    "clothing", "shoes", "handbag",
    "software", "app", "download",
]

# ── Comment-video linking: category keywords ───────────────────────────────────
LINK_CATEGORY_KEYWORDS = {
    "camera_optics": [
        "camera", "lens", "drone", "gopro", "photography", "gimbal",
        "tripod", "stabilizer", "mirrorless", "dslr", "action cam",
    ],
    "digital_accessories": [
        "cable", "charger", "power bank", "ssd", "hard drive", "usb",
        "storage", "adapter", "hub", "dock", "laptop stand", "mouse",
    ],
    "gaming": [
        "controller", "keyboard", "mouse", "headset", "gaming",
        "playstation", "xbox", "nintendo", "steam deck", "switch",
    ],
    "travel_outdoor": [
        "travel", "camping", "hiking", "backpack", "luggage",
        "outdoor", "adventure", "packing", "gear bag", "duffel",
    ],
    "collectibles": [
        "watch", "jewelry", "collectible", "figurine", "statue",
        "arcade", "model", "vinyl", "lego",
    ],
    "tools_equipment": [
        "tool", "drill", "saw", "precision", "repair", "hardware",
        "mechanical", "instrument",
    ],
    "beauty_medical": [
        "makeup", "beauty", "skincare", "medical", "health",
        "cosmetics", "brush", "dermatology",
    ],
    "audio": [
        "speaker", "microphone", "audio", "headphone", "earphone",
        "airpod", "soundbar", "amplifier", "synthesizer",
    ],
    "general_protection": [
        "case", "cover", "bag", "pouch", "sleeve", "organizer",
        "storage", "protection", "protective",
    ],
}

# ── Data cleaning: regex patterns ─────────────────────────────────────────────
CLEAN_EXCLUDED_TITLE_PATTERNS = [
    r"\biphone\b", r"\bphone case\b", r"\bsmartphone\b",
    r"\biPad\b", r"\btablet case\b",
    r"\bfashion case\b", r"\bcute case\b", r"\baesthetic case\b",
    r"\bskincare\b", r"\bmakeup\b",
]

CLEAN_RELEVANT_CATEGORIES = [
    "camera_optics", "digital_accessories", "gaming",
    "travel_outdoor", "collectibles", "tools_equipment",
    "audio", "general_protection",
]

CLEAN_LOW_VALUE_PATTERNS = [
    r"^\s*(lol|lmao|wow|omg|wtf|haha|hahaha|hey|hi|ok|okay|yes|no|thanks?|thx)\s*$",
    r"^(first|second|third)\s*$",
]

CLEAN_PRODUCT_KEYWORDS = [
    "case", "cover", "protect", "protection", "damaged", "damage",
    "drop", "scratch", "break", "broken", "crack",
    "durable", "sturdy", "flimsy", "cheap", "quality",
    "storage", "carry", "bag", "box", "pouch", "sleeve",
    "packaging", "unboxing", "contents", "included",
    "plastic", "metal", "aluminum", "leather", "foam", "padding",
    "rubber", "nylon", "fabric", "hard", "soft",
    "zipper", "magnetic", "clip", "strap", "handle", "pocket",
    "compartment", "waterproof", "shockproof", "lightweight",
    "fits", "size", "big", "small", "roomy", "spacious",
    "love", "hate", "recommend", "worth", "price", "value",
    "great", "terrible", "amazing", "disappointed", "impressed",
]

# ── Demand signal detection patterns ───────────────────────────────────────────
DEMAND_STRONG_PATTERNS = {
    "purchase_intent": [
        r"\bneed\b.*\bcase\b", r"\bneed\b.*\bprotect",
        r"\blooking\b.*\bcase\b", r"\blooking\b.*\bfor\b.*\bprotect",
        r"\bsearching\b.*\bcase\b", r"\bwish\b.*\bhad\b.*\bcase",
        r"\bbuying\b.*\bcase\b", r"\bbought\b.*\bcase\b",
        r"\bjust\s*bought\b", r"\bjust\s*ordered\b", r"\bjust\s*got\b",
        r"can'?t\s*find\b.*\bcase\b", r"nowhere\s*to\s*find\b",
        r"\brecommend\b.*\bcase\b", r"\bany\s+suggestions\b.*\bcase\b",
        r"\bbest\s+case\b.*\bfor\b", r"\bcase\s+recommendation\b",
        r"\bdoes\s+it\s+fit\b.*\bcase\b", r"\bwhat\s+case\s+fits\b",
    ],
    "problem_complaint": [
        r"\bbroke\b", r"\bbroken\b", r"\bdamaged\b", r"\bdamage\b",
        r"\bcracked\b", r"\bdropped\b.*\bbroke\b",
        r"\bdoesn'?t\s+protect\b", r"\bdoesn'?t\s+fit\b",
        r"\bcheap\b", r"\bflimsy\b", r"\bpoor\s+quality\b",
        r"\bterrible\b", r"\bworst\b", r"\bdisappointed\b",
        r"\btoo\s+small\b", r"\btoo\s+big\b",
        r"\bneed\s+more\b", r"\bneed\s+better\b",
        r"\bshould\s+have\b.*\bcase\b",
    ],
    "storage_travel": [
        r"\bhow\s+to\s+pack\b", r"\bpack\s+my\b",
        r"\bpacking\s+list\b", r"\bstorage\s+solution\b",
        r"\borganize\b", r"\borganizer\b",
        r"\bcarry\s+case\b", r"\bcarrying\s+case\b",
        r"\btravel\s+case\b", r"\bEDC\b", r"\beveryday\s+carry\b",
        r"\bportable\b.*\bsolution\b",
        r"\bshockproof\b", r"\bwaterproof\b",
        r"\bhard\s+case\b", r"\bhardshell\b",
    ],
    "review_comparison": [
        r"\bvs\b", r"\bversus\b", r"\bbetter\s+than\b",
        r"\bcompare\s+to\b", r"\binstead\s+of\b",
        r"\breview\b", r"\bunboxing\b",
        r"\bafter\s+using\b", r"\brecommend\b",
        r"\bworth\s+it\b", r"\bprice\s+for\b",
    ],
    "general_question": [
        r"\bhow\s+does\b", r"\bwhat\s+is\b", r"\bdoes\s+it\b",
        r"\bcan\s+I\b", r"\bwill\s+it\b",
        r"\bwhere\s+to\s+buy\b", r"\bhow\s+much\b",
    ],
}

DEMAND_EXCLUDE_PATTERNS = {
    "nonsense": [
        r"^\s*(lol|lmao|rofl|wth|omg|wtf|haha|hahaha)\s*$",
        r"^\s*(first|second|third)\s*$",
        r"^\s*(yes|no|ok|okay|yeah)\s*$",
        r"^\s*(hi|hey|yo|sup)\s*$",
        r"^\s*(thanks?|thx|ty)\s*$",
        r"^\s*(nice|cool|awesome|amazing|wow)\s*$",
        r"^\s*(subscribe|like|comment)\s*$",
    ],
    "irrelevant": [
        r"\bphone\s+case\b", r"\biphone\s+case\b",
        r"\bsamsung\s+case\b", r"\biPad\s+case\b",
        r"\bsoftware\b", r"\bdownload\b", r"\bapp\b",
    ],
}

# ── LLM classification labels ──────────────────────────────────────────────────
DEMAND_SIGNAL_LABELS = [
    "purchase_intent",
    "problem_complaint",
    "comparison_research",
    "usage_scenario",
    "wishful_thinking",
    "supply_recommendation",
    "no_signal",
]

LLM_SYSTEM_PROMPT = """You are a consumer product demand signal analyst.

Task: Read each YouTube comment and determine whether it reveals a demand signal for a PROFESSIONAL STORAGE or PROTECTIVE PRODUCT.

[Important] Exclusion scope (do NOT treat the following categories as demand signals):
- Phone cases, phone covers, phone screen protectors, phone protective cases
- Tablet cases, tablet protective covers
- Earbud cases, earbud covers, AirPods protective cases
- Smartwatch bands, watch cases
- Laptop sleeves, laptop inner bags (lightweight protection used as accessories only)
- Any generic small accessories named "case", "cover", "shell", or "sleeve" that are merely auxiliary protective accessories for electronic devices

[Inclusion scope] The following categories belong to the target products — please keep them in scope:
- Hard cases, soft pouches, storage bags, camera bags, tool boxes
- Instrument cases, DJ equipment cases, travel organizers, outdoor gear bags
- Professional protective cases (such as Pelican, Nanuk brand heavy-duty protective cases)
- Industrial or professional storage containers, equipment cases, instrument cases
- Drone-specific cases, lens cases, photography equipment cases
- Any storage or protective container with independent use value, relatively sturdy structure, or specialized storage or protective containers

Judgment logic: Focus on whether a "protective case or storage container" is the core subject of the comment, or whether the user has a clear functional need for it (e.g., protection, storage, portability, durability). If a comment only mentions a protective case incidentally while discussing the main electronic product, or brushes over it in passing, it does NOT count as a demand signal.

Classification labels:

  1. purchase_intent — Explicitly expresses a desire or plan to purchase a professional storage or protective product.
     Examples:
       "I need a hard case for my drone"
       "Looking for a storage pouch that can protect my camera"
       "Just bought a new lens, need a case to go with it"
     Excluded examples (phone cases etc. are out of scope):
       "I want a phone case"
       "Any recommended tablet protective covers?"

  2. problem_complaint — Expresses dissatisfaction about equipment damage or lack of protection.
     Examples:
       "My equipment keeps getting scratched"
       "The foam padding is terrible quality, items arrived damaged"
       "My lens broke from a fall, should have bought a better protective case"

  3. comparison_research — Actively compares or researches different storage or protective products.
     Examples:
       "Is a hard case or a soft pouch better for outdoor shooting?"
       "Which is more durable, Nanuk or Pelican?"
       "What kind of case is best for drones?"
     Excluded examples:
       "Which phone case brand is better?"

  4. usage_scenario — Describes a specific scenario where protection or storage is needed.
     Examples:
       "I travel with my gear a lot and need shockproof storage"
       "This case is perfect for hiking and outdoor adventures"
       "I use this storage bag to organize my everyday carry items"

  5. wishful_thinking — Regrets not purchasing a storage or protective product.
     Examples:
       "I should have gotten a hard case for my Switch earlier"
       "Regret not buying the professional version"
       "It would be nice if it came with a storage bag"
     Excluded examples:
       "I should have just bought a phone case"

  6. supply_recommendation — Recommends or positively reviews a specific professional storage or protective product.
     Examples:
       "This hard case protected my camera from a serious fall"
       "Best protective case I have ever used, highly recommend"
       "The foam insert is perfectly cut, all my gear fits in it"
     Excluded examples:
       "This phone case is really good"

  7. no_signal — Does NOT reveal demand for professional storage or protective products.
     Examples:
       "Great video, thanks for sharing"
       "Haha so funny"
       "Can you make a video about XX instead?"
       "What brand is the camera in this video?"

Output format: Return a JSON array, one object per comment.

{
  "results": [
    {
      "comment_id": "<comment_id>",
      "signal": "<label>",
      "confidence": <0.0-1.0>,
      "reason": "<2-4 sentences explaining WHY this comment is or is not a demand signal>"
    }
  ]
}

Rules:
- Comments may be in any language — analyze all of them.
- If difficult to judge, choose the closest label and explain.
- confidence is your self-assessed certainty from 0.0 to 1.0.
- reason must specifically call out the key content in the comment that triggered the label.
- Output valid JSON only. No markdown, no additional explanation.
"""
