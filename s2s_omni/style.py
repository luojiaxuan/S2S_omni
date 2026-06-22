from __future__ import annotations

import re


HIGH_RISK_ZH_PHRASES = [
    "足证",
    "他熊",
    "极有趣",
    "深研",
    "此拟声",
    "其拟声",
    "或为偶然",
    "岂不",
    "遇一小童",
    "小童",
    "称将即刻",
    "将即刻",
    "以指覆目",
    "半阖",
    "众酋长",
    "主酋长",
    "尔等",
    "尔辈",
    "吾",
    "汝",
]

BOOKISH_ZH_TERMS = [
    "乃",
    "遂",
    "亦",
    "皆",
    "若",
    "尚",
    "即刻",
    "未果",
    "定会",
    "不明所以",
    "以为和平",
    "褫夺",
]

MIXED_CJK_LATIN_RE = re.compile(r"[\u4e00-\u9fff][A-Z]{2,}|[A-Z]{2,}[\u4e00-\u9fff]")
CLASSICAL_ZH_RE = re.compile(r"视.{1,8}如(?:兄弟|己出|珍宝|草芥)")


def style_violations(text: str, lang: str) -> list[str]:
    if not lang.startswith("zh"):
        return []
    reasons: list[str] = []
    high_risk_hits = [term for term in HIGH_RISK_ZH_PHRASES if term in text]
    bookish_hits = [term for term in BOOKISH_ZH_TERMS if term in text]
    if high_risk_hits:
        reasons.append("archaic_or_classical_style:" + ",".join(high_risk_hits))
    elif CLASSICAL_ZH_RE.search(text):
        reasons.append("archaic_or_classical_style:视X如Y")
    elif len(bookish_hits) >= 2:
        reasons.append("bookish_written_style:" + ",".join(bookish_hits))
    if MIXED_CJK_LATIN_RE.search(text):
        reasons.append("mixed_cjk_latin_name")
    return reasons
