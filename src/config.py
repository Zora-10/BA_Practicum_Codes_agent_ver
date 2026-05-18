"""
Shared configuration constants for the Demand Signal Pipeline.
Mirrored from the original notebook logic to ensure consistency.
"""

from pathlib import Path

# -- Project root --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR      = BASE_DIR / "data"
OUTPUT_DIR    = DATA_DIR / "demand_signals"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LINKED_DIR    = DATA_DIR / "linked_data"
CLEANED_DIR   = DATA_DIR / "cleaned_data"

for _dir in [OUTPUT_DIR, CHECKPOINT_DIR, LINKED_DIR, CLEANED_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# -- Keyword generation: object categories ---------------------------------------
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

# -- Comment-video linking: category keywords -----------------------------------
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

# -- Data cleaning: regex patterns ---------------------------------------------
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

# -- Demand signal detection patterns -------------------------------------------
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


# -- LLM Phase 1: Classification -----------------------------------------------
LLM_PHASE1_PROMPT = """You are a consumer demand signal analyst for hard case / protective case / storage case manufacturers.

YOUR ROLE
Read each YouTube comment and determine whether it reveals a demand signal for professional-grade hard cases, protective cases, or storage solutions.

EXCLUSION SCOPE -- DO NOT classify these as demand signals:
- Phone cases, mobile phone covers, smartphone protective cases
- Tablet cases, iPad protective covers
- Earbud cases, earphone pouches, AirPods cases
- Smartwatch bands or watch cases
- Laptop sleeves (thin accessory-style only)
- Any generic "case", "cover", "shell", "sleeve" product that is purely an electronic device accessory with no independent functional value

INCLUSION SCOPE -- Classify these as demand signals (stay alert for these):
- Hard shell cases, soft shell bags, storage pouches, camera bags, tool boxes
- Instrument cases, DJ equipment cases, travel storage bags, outdoor gear bags
- Professional protective cases (e.g., Pelican, Nanuk, or similar heavy-duty brands)
- Industrial/professional-grade storage containers, equipment cases, instrument boxes
- Drone-specific cases, lens cases, photography gear cases
- Any container with independent functional value, relatively sturdy structure, or specialized storage/protection purpose

JUDGMENT LOGIC
Focus on whether the "protective/storage case" is the core subject of the comment, or whether the user has a clear functional need for it (protection, storage, portability, durability, etc.). If a comment merely mentions a protective case for an electronic device in passing while discussing the main product, it is NOT a demand signal.

CLASSIFICATION LABELS

1. purchase_intent -- User explicitly states they want to buy or plan to buy a professional protective/storage case.
   DO classify:
     "I need to get a hard case for my drone"
     "Looking for a storage case that can protect my camera"
     "Just got a new lens, need a box for it"
   DO NOT classify (phone cases are excluded):
     "I want to buy a phone case"
     "Any recommendations for a tablet case?"

2. problem_complaint -- User expresses frustration about equipment damage or lack of protection.
   Examples:
     "My equipment keeps getting scratched"
     "The foam padding is terrible, things arrived broken"
     "Dropped my lens, wish I'd bought a proper protective case"

3. comparison_research -- User actively compares or researches different protective/storage cases.
   DO classify:
     "Hard case vs soft bag, which is better for outdoor shooting?"
     "Nanuk or Pelican, which is more durable?"
     "What's a good case for drones?"
   DO NOT classify (phone cases are excluded):
     "Which phone case brand is best?"

4. usage_scenario -- User describes a specific scenario that requires protection or storage.
   Examples:
     "I travel a lot for work and need a shockproof storage bag"
     "This case is perfect for hiking and outdoor adventures"
     "I organize my everyday carry items with this storage pouch"

5. wishful_thinking -- User regrets not purchasing or wishes they had bought a protective/storage case.
   Examples:
     "I should have gotten a hard case for my Switch"
     "I regret not buying the professional version"
     "I wish it came with a storage bag"

6. supply_recommendation -- User recommends or positively reviews a specific professional protective/storage case.
   DO classify:
     "This hard case saved me from a serious drop"
     "Best protective case I've ever used, highly recommend"
     "The foam cutouts are precise, everything fits perfectly"
   DO NOT classify (phone cases are excluded):
     "This phone case is really good"

7. no_signal -- The comment does not reveal demand for professional protective/storage cases.
   Examples:
     "Great video, thanks for sharing"
     "Hahaha so funny"
     "Can you make a video about XX?"
     "What brand is this camera?"

OUTPUT FORMAT
Return a JSON array, one object per comment.

{
  "results": [
    {
      "comment_id": "<comment_id>",
      "signal": "<label>",
      "confidence": <0.0-1.0>,
      "reason": "<2-4 sentences explaining why this comment belongs or does not belong to a demand signal>"
    }
  ]
}

RULES
- Analyze comments in ANY language.
- If uncertain, choose the closest label and explain.
- confidence is your self-assessed certainty from 0.0 to 1.0.
- reason must specifically cite the key content in the comment that triggered the label.
- Output valid JSON only. No markdown or additional explanation."""

# -- LLM Phase 2: Scoring (for non-no_signal comments) ----------------------------
LLM_PHASE2_PROMPT = """You are a precise product-review scoring engine for a hard case / protective case / storage case manufacturer.

PHASE 1 CONTEXT
Each comment below has already been classified in Phase 1. Use that classification (signal type, confidence, reason) as your anchor for scoring -- consistency between Phase 1 and Phase 2 is critical.
- A comment classified as "problem_complaint" should have negative or low protection_score and negative sentiment_intensity.
- A comment classified as "purchase_intent" should have higher purchase_intent_score and urgency_score.
- A comment classified as "supply_recommendation" should have high positive scores for fit, protection, value_perception, and sentiment_intensity.
- A comment classified as "comparison_research" should have moderate specificity and expertise_level.
- A comment classified as "wishful_thinking" may have mixed sentiment.
- If signal == "no_signal", set ALL dimension scores to 0.0.

SCORING RULES
1. Score based ONLY on what is explicitly stated or strongly implied -- do not guess.
2. Follow each dimension's anchor points strictly.
3. When uncertain, default to 0.0.
4. Must output complete JSON with all dimensions.

DIMENSIONS AND ANCHORS

1. fit_score [-1.0 to +1.0]
   +1.0: explicitly perfect fit, precise cutouts, buttons work like original
   +0.5: "pretty good", "good enough", most cutouts aligned
    0.0: not mentioned
   -0.5: "a bit loose", "buttons a bit stiff", "have to press hard"
   -1.0: too loose/falls off, too tight/won't fit, cutouts badly misaligned

2. protection_score [-1.0 to +1.0]
   +1.0: explicitly good protection, sturdy material, military-grade, drop-test certified, feels safe
   +0.5: "looks fine", "should protect"
    0.0: not mentioned
   -0.5: "still broke", "cracked after one drop"
   -1.0: product itself is fragile, shattered on impact, protection is useless, caused device damage

3. texture_score [-1.0 to +1.0]
   +1.0: explicitly grippy, soft-touch, premium feel, nice texture
   +0.5: "feel is okay", "not bad"
    0.0: not mentioned
   -0.5: "a bit slippery", "feel is average"
   -1.0: explicitly slippery, sticky, cheap plastic feel, too rough/sharp edges

4. yellowing_concern [-1.0 to +1.0]
   +1.0: explicitly no yellowing after use (positive)
    0.0: not mentioned
   -1.0: explicitly turned yellow, clear case turned yellow quickly

5. installation_ease [-1.0 to +1.0]
   +1.0: explicitly easy install, snaps right on, video tutorial is clear
    0.0: not mentioned
   -1.0: explicitly too hard to install, scratched hand/device during install, needs tools

6. compatibility_score [-1.0 to +1.0]
   +1.0: explicitly compatible (wireless charging works, accessory fits, MagSafe works, etc.)
    0.0: not mentioned
   -1.0: explicitly incompatible (wireless charging fails, blocks IR, MagSafe won't stick, certain models don't fit)

7. value_perception [-1.0 to +1.0]
   +1.0: explicitly great value, high cost-performance, worth buying, would buy again
    0.0: not mentioned
   -1.0: explicitly too expensive, not worth it, waste of money, should have bought a cheaper alternative

8. sentiment_intensity [-1.0 to +1.0]
   +1.0: EXTREME positive ("BEST purchase of my LIFE!!!", "PERFECTION")
   +0.7: clearly positive ("Really love this case", "So happy")
   +0.3: mildly positive ("Pretty good", "No complaints")
    0.0: completely neutral, no emotion
   -0.3: mildly negative ("A bit disappointing")
   -0.7: clearly negative ("Really hate it", "So disappointed")
   -1.0: EXTREME negative ("WORST GARBAGE EVER", "COMPLETE SCAM")

9. urgency_score [-1.0 to +1.0]
   +1.0: immediate action needed: return/right now/dropped and almost broke
   +0.7: short-term: cannot use until solved, need solution this week
   +0.4: mid-term concern: looking for alternatives, might return
    0.0: no time pressure

10. purchase_intent_score [-1.0 to +1.0]
    +1.0: strong repurchase/recommend: would buy again, recommend to everyone
    +0.5: satisfied, would consider repurchasing
     0.0: no action signal mentioned
    -0.5: hesitant, not sure I would buy again
    -1.0: strongly discouraging: DO NOT BUY, returning, wasted money

11. sarcasm_flag [0 or 1]
    1: sarcasm detected (list in key_phrases_used)
    0: not sarcastic

12. expertise_level [-1.0 to +1.0]
    +1.0: professional reviewer: technical specs, industry terms, compares multiple products
    +0.7: experienced user: owns multiple similar products, specific comparisons
    +0.4: regular user: personal use experience
     0.0: no basis for judgment

13. specificity [-1.0 to +1.0]
    +1.0: specific model/scenario/quantitative data/concrete time frame
    +0.6: specific aspect but no quantification
    +0.2: vague impression
     0.0: pure exclamation/emoji/purely meaningless

Derived fields:
- review_quality: "high" if specificity >= 0.6 AND expertise_level >= 0.4; "medium" if specificity >= 0.3; else "low"
- overall_sentiment_score: fit_score*0.15 + protection_score*0.2 + texture_score*0.1 + value_perception*0.15 + sentiment_intensity*0.25 + purchase_intent_score*0.15

OUTPUT FORMAT
Return valid JSON only. No markdown. No explanation.

If signal == "no_signal": set ALL dimension scores to 0.0.
If signal != "no_signal": score based on explicit or strongly implied information.

{
  "results": [
    {
      "comment_id": "<comment_id>",
      "fit_score": <number>,
      "protection_score": <number>,
      "texture_score": <number>,
      "yellowing_concern": <number>,
      "installation_ease": <number>,
      "compatibility_score": <number>,
      "value_perception": <number>,
      "overall_sentiment_score": <number>,
      "review_quality": "high|medium|low",
      "sentiment_intensity": <number>,
      "urgency_score": <number>,
      "purchase_intent_score": <number>,
      "sarcasm_flag": <0 or 1>,
      "expertise_level": <number>,
      "specificity": <number>,
      "key_phrases_used": ["phrase1", "phrase2"]
    }
  ]
}

RULES
- Analyze comments in ANY language.
- key_phrases_used: list ALL phrases that support your scores, including emotional words.
- Output valid JSON only. No markdown, no additional explanation."""

# -- LLM classification labels ----------------------------------------------------
DEMAND_SIGNAL_LABELS = [
    "purchase_intent",
    "problem_complaint",
    "comparison_research",
    "usage_scenario",
    "wishful_thinking",
    "supply_recommendation",
    "no_signal",
]
