"""
Section map: maps each PDF page number to its section, annexure, and policy title.
Built from the Table of Contents on pages 2-3 of the HR Manual.

The TOC lists:
  Section 1 (p5): Employing Staff
  Section 2 (p16): Performance & Discipline
  Section 3 (p22): Grievance & Whistleblowing
  Section 4 (p28): Equality & Health/Safety
  Company Rules and Regulation (p34)
  Annexure I (p39): On boarding, Probation, Training, Relocation, Off boarding
  Annexure II (p48): Equal Opportunities, Data card/Visiting Cards/Emails
  Annexure III (p52): Dress code, Work timing, Punctuality, Compensation,
                       Staff advances, Employment of Relatives, Leave policy,
                       Reward & Recognition, Whistle Blower & Anti Sexual Harassment
  Annexure IV (p66): Long Distance Travel, Short Distance Travel
"""

from typing import Dict, Tuple

# Each entry: page_number -> (section_label, policy_title)
# page_number is the 1-indexed PDF page (matches footer "HR Manual - N").
_SECTION_MAP: Dict[int, Tuple[str, str]] = {}


def _fill_range(start: int, end: int, section: str, policy: str) -> None:
    """Helper: fill page range [start, end] inclusive."""
    for p in range(start, end + 1):
        _SECTION_MAP[p] = (section, policy)


# ── Cover & TOC ─────────────────────────────────────────────────────
_fill_range(1, 1, "Cover", "HR Policy Manual")
_fill_range(2, 3, "Table of Contents", "Table of Contents")
_fill_range(4, 4, "Introduction", "Introduction")

# ── Section 1: Employing Staff ──────────────────────────────────────
_fill_range(5, 5, "Section 1", "Employing Staff")
_fill_range(6, 7, "Section 1", "1.1 Recruitment / 1.2 Induction Policy")
_fill_range(8, 8, "Section 1", "1.3 Employee's Employment")
_fill_range(9, 11, "Section 1", "1.4 Standards of Business Conduct")
_fill_range(12, 15, "Section 1", "1.5 Internet and Email Usage Policy")

# ── Section 2: Performance & Discipline ─────────────────────────────
_fill_range(16, 16, "Section 2", "Performance and Discipline")
_fill_range(17, 17, "Section 2", "2.1 Performance Appraisal Policy")
_fill_range(18, 18, "Section 2", "2.2 Controlling Absence Policy")
_fill_range(19, 20, "Section 2", "2.3 Disciplinary Procedure")
_fill_range(21, 21, "Section 2", "2.4 Capability Procedure")

# ── Section 3: Grievance & Whistleblowing ───────────────────────────
_fill_range(22, 22, "Section 3", "Grievance and Whistleblowing")
_fill_range(23, 24, "Section 3", "3.1 Grievance Procedure")
_fill_range(25, 25, "Section 3", "3.2 Prevention of Bullying and Harassment")
_fill_range(26, 27, "Section 3", "3.3 Whistleblowing Policy")

# ── Section 4: Equality & Health/Safety ─────────────────────────────
_fill_range(28, 28, "Section 4", "Equality and Health/Safety")
_fill_range(29, 30, "Section 4", "4.1 Equality Policy")
_fill_range(31, 31, "Section 4", "4.2 Drug and Alcohol Abuse Policy")
_fill_range(31, 33, "Section 4", "4.3 Health and Safety Policy")

# ── Company Rules and Regulation ────────────────────────────────────
_fill_range(34, 38, "Company Rules & Regulation", "Company Rules and Regulation")

# ── Annexure I ──────────────────────────────────────────────────────
_fill_range(39, 39, "Annexure I", "Annexure I Cover")
_fill_range(40, 41, "Annexure I", "[A] On Boarding Process")
_fill_range(42, 43, "Annexure I", "[B] Probation Policy")
_fill_range(44, 44, "Annexure I", "[C] Training and Development")
_fill_range(45, 45, "Annexure I", "[D] Relocation Policy")
_fill_range(46, 47, "Annexure I", "[E] Off Boarding Policy")

# ── Annexure II ─────────────────────────────────────────────────────
_fill_range(48, 49, "Annexure II", "Annexure II Cover")
_fill_range(50, 50, "Annexure II", "[A] Equal Opportunities")
_fill_range(51, 51, "Annexure II", "[B] Data Card/Visiting Cards/Emails Policy")

# ── Annexure III ────────────────────────────────────────────────────
_fill_range(52, 52, "Annexure III", "Annexure III Cover")
_fill_range(53, 53, "Annexure III", "[A] Dress Code")
_fill_range(54, 54, "Annexure III", "[B] Work Timing Policy")
_fill_range(55, 56, "Annexure III", "[C] Punctuality and Attendance")
_fill_range(57, 57, "Annexure III", "[D] Compensation Policy")
_fill_range(58, 58, "Annexure III", "[E] Staff Advances")
_fill_range(59, 59, "Annexure III", "[F] Employment of Relatives Policy")
_fill_range(60, 62, "Annexure III", "[G] Leave Policy")
_fill_range(63, 63, "Annexure III", "[H] Reward & Recognition")
_fill_range(64, 65, "Annexure III", "[I] Whistle Blower Policy & Anti Sexual Harassment Policy")

# ── Annexure IV ─────────────────────────────────────────────────────
_fill_range(66, 66, "Annexure IV", "Annexure IV Cover")
_fill_range(67, 72, "Annexure IV", "[A] Long Distance Travel Policy")
_fill_range(73, 79, "Annexure IV", "[B] Short Distance Travel Policy")


def get_section_info(page_num: int) -> Tuple[str, str]:
    """
    Return (section_label, policy_title) for a given 1-indexed page number.
    Falls back to ("Unknown", "Unknown") for unmapped pages.
    """
    return _SECTION_MAP.get(page_num, ("Unknown", "Unknown"))


def get_metadata_for_page(page_num: int) -> dict:
    """
    Return a metadata dict for a given page number, suitable for chunk metadata.
    """
    section, policy = get_section_info(page_num)
    return {
        "page": page_num,
        "section": section,
        "policy_title": policy,
        "source": "hr_manual.pdf",
    }
