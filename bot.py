from __future__ import annotations

import os
import re
import time
import json
import hashlib
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import request as urlrequest
from urllib import error as urlerror

from fastapi import FastAPI, Request

app = FastAPI(title="Vera Signal-to-Action Engine", version="2.0.0")
START_TIME = time.time()

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

# (scope, context_id) -> {version, payload}
CONTEXTS: dict[tuple[str, str], dict[str, Any]] = {}

# conversation_id -> state
CONVERSATIONS: dict[str, dict[str, Any]] = {}

# merchant_id -> recent inbound fingerprints / timestamps
MERCHANT_REPLY_HISTORY: dict[str, deque[tuple[float, str]]] = defaultdict(lambda: deque(maxlen=20))

# Merchant-level send memory. Kept small on purpose.
MERCHANT_SEND_MEMORY: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=20))

# trigger IDs already sent once (global dedup)
SENT_TRIGGER_IDS: set[str] = set()

INTERNAL_JARGON = {
    "suppression_key", "trigger_id", "trigger context", "merchantcontext",
    "customercontext", "categorycontext", "payload", "embedding", "vector store",
    "state machine", "decision engine", "router", "json"
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def pct_abs(value: Any) -> Optional[str]:
    try:
        x = abs(float(value))
    except (ValueError, TypeError):
        return None
    return f"{x * 100:.0f}%" if x <= 1 else f"{x:.0f}%"


def pct(value: Any) -> Optional[str]:
    try:
        x = float(value)
    except (ValueError, TypeError):
        return None
    return f"{x * 100:.0f}%" if abs(x) <= 1 else f"{x:.0f}%"


def money(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    s = str(value)
    if "₹" in s:
        return s
    # Preserve non-price strings such as free_delivery.
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        if "." in s:
            return f"₹{s}"
        return f"₹{int(float(s)):,}"
    return s


def find_by_id(items: list[dict[str, Any]], item_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not item_id:
        return None
    for item in items or []:
        if item.get("id") == item_id:
            return item
    return None


def active_offers(merchant: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for o in merchant.get("offers", []) if o.get("status") == "active"]


def first_active_offer(merchant: dict[str, Any], keywords: list[str] | None = None) -> Optional[dict[str, Any]]:
    offers = active_offers(merchant)
    if not keywords:
        return offers[0] if offers else None
    kws = [k.lower() for k in keywords]
    for offer in offers:
        title = offer.get("title", "").lower()
        if any(k in title for k in kws):
            return offer
    return offers[0] if offers else None


def customer_safe_offer(merchant: dict[str, Any], customer: dict[str, Any] | None) -> Optional[dict[str, Any]]:
    offers = active_offers(merchant)
    if not customer:
        return offers[0] if offers else None
    state = customer.get("state", "active")
    acceptable = {"all", "repeat_user"} if state in {"active", "lapsed_soft", "lapsed_hard", "churned"} else {"all", "new_user", "repeat_user", "senior"}
    for offer in offers:
        aud = offer.get("audience")
        if aud is None or aud in acceptable:
            return offer
    return None


def owner_name(merchant: dict[str, Any]) -> str:
    return merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name") or "there"


def merchant_name(merchant: dict[str, Any]) -> str:
    return merchant.get("identity", {}).get("name") or owner_name(merchant)


def category_voice(category: dict[str, Any]) -> dict[str, Any]:
    return category.get("voice", {}) or {}


def trigger_strength(kind: str) -> float:
    weights = {
        "recall_due": 10, "appointment_tomorrow": 10, "chronic_refill_due": 10,
        "supply_alert": 10, "regulation_change": 10, "customer_lapsed_hard": 9,
        "customer_lapsed_soft": 8, "trial_followup": 8, "wedding_package_followup": 8,
        "active_planning_intent": 9, "perf_dip": 8, "seasonal_perf_dip": 8,
        "competitor_opened": 7, "gbp_unverified": 7, "renewal_due": 7,
        "cde_opportunity": 6, "category_seasonal": 6, "festival_upcoming": 5,
        "perf_spike": 5, "milestone_reached": 5, "review_theme_emerged": 6,
        "winback_eligible": 6, "curious_ask_due": 4, "dormant_with_vera": 3,
        "research_digest": 4, "ipl_match_today": 8, "category_research_digest_release": 4,
    }
    return weights.get(kind, 3)


def evidence_values(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any], customer: dict[str, Any] | None) -> set[str]:
    """Collect literal facts we allow a polished message to contain."""
    vals: set[str] = set()

    def add(v: Any):
        if v is None:
            return
        s = str(v).strip()
        if s:
            vals.add(s)

    mi = merchant.get("identity", {})
    add(mi.get("name")); add(mi.get("owner_first_name")); add(mi.get("city")); add(mi.get("locality"))
    p = merchant.get("performance", {})
    for k in ["views", "calls", "directions", "leads", "ctr"]:
        add(p.get(k))
    for k, v in (p.get("delta_7d") or {}).items():
        add(v); add(pct(v))
    for o in active_offers(merchant):
        add(o.get("title")); add(o.get("id"))
    for s in merchant.get("signals", []): add(s)
    for k, v in (trigger.get("payload") or {}).items():
        if isinstance(v, (str, int, float)):
            add(v); add(pct(v))
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, (str, int, float)): add(x)
                elif isinstance(x, dict):
                    for y in x.values():
                        if isinstance(y, (str, int, float)): add(y)
    if customer:
        ci = customer.get("identity", {})
        add(ci.get("name")); add(ci.get("language_pref"))
        rel = customer.get("relationship", {})
        for k, v in rel.items():
            if isinstance(v, (str, int, float)): add(v)
            elif isinstance(v, list):
                for x in v: add(x)
        prefs = customer.get("preferences", {})
        for v in prefs.values(): add(v)
        con = customer.get("consent", {})
        for v in con.values():
            if isinstance(v, list):
                for x in v: add(x)
            else: add(v)

    # Category facts are available to the composer but should not all leak into copy.
    peer = category.get("peer_stats", {}) or {}
    for k in ["avg_rating", "avg_review_count", "avg_views_30d", "avg_calls_30d", "avg_ctr", "retention_6mo_pct", "monthly_churn_pct", "trial_to_paid_pct"]:
        add(peer.get(k)); add(pct(peer.get(k)))
    return vals


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])\d+(?:[\.,]\d+)?%?", text or ""))


def evidence_number_tokens(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any], customer: dict[str, Any] | None) -> set[str]:
    allowed = set()

    def add(v: Any):
        if v is None:
            return
        s = str(v)
        for m in re.findall(r"\d+(?:[\.,]\d+)?%?", s):
            allowed.add(m)
        pv = pct(v)
        if pv:
            for m in re.findall(r"\d+(?:[\.,]\d+)?%?", pv):
                allowed.add(m)

    for v in evidence_values(category, merchant, trigger, customer): add(v)
    return allowed


def has_consent_for(customer: dict[str, Any] | None, trigger_kind: str) -> bool:
    if not customer:
        return True
    prefs = customer.get("preferences", {}) or {}
    channel = prefs.get("channel")
    if channel not in (None, "whatsapp") and "whatsapp" not in str(channel).lower():
        return False
    if prefs.get("reminder_opt_in") is False:
        # Promotional triggers should definitely be blocked; operational reminders may
        # still be allowed only if explicit scope includes the event.
        if trigger_kind not in {"recall_due", "appointment_tomorrow", "chronic_refill_due", "trial_followup", "wedding_package_followup"}:
            return False
    scope = set(customer.get("consent", {}).get("scope", []) or [])
    mapping = {
        "recall_due": {"recall_reminders", "appointment_reminders"},
        "appointment_tomorrow": {"appointment_reminders"},
        "chronic_refill_due": {"refill_reminders", "appointment_reminders"},
        "trial_followup": {"appointment_reminders", "promotional_offers"},
        "wedding_package_followup": {"bridal_package_followup", "appointment_reminders"},
        "customer_lapsed_soft": {"promotional_offers", "winback_offers"},
        "customer_lapsed_hard": {"promotional_offers", "winback_offers"},
    }
    required = mapping.get(trigger_kind)
    if required and scope and not (required & scope):
        # Synthetic generated customers often carry only promotional_offers while
        # still having reminder_opt_in=true. Treat operational reminders as allowed,
        # but keep promotional win-back constrained by its explicit scope.
        operational = {"recall_due", "appointment_tomorrow", "chronic_refill_due"}
        if trigger_kind in operational and prefs.get("reminder_opt_in") is True:
            return True
        return False
    return True


def salutation(category: dict[str, Any], merchant: dict[str, Any], customer: dict[str, Any] | None) -> str:
    if customer:
        name = customer.get("identity", {}).get("name", "there")
        if merchant.get("category_slug") == "dentists":
            return f"Hi {name}"
        return f"Hi {name}"
    first = owner_name(merchant)
    slug = merchant.get("category_slug") or category.get("slug")
    if slug == "dentists":
        return f"Dr. {first.replace('Dr. ', '')}"
    return first


def best_digest_item(category: dict[str, Any], trigger: dict[str, Any]) -> Optional[dict[str, Any]]:
    payload = trigger.get("payload", {}) or {}
    item = find_by_id(category.get("digest", []) or [], payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id"))
    if item:
        return item
    # Search by obvious trigger payload identifiers.
    for i in category.get("digest", []) or []:
        blob = json.dumps(i, ensure_ascii=False).lower()
        for key in ["title", "kind", "id"]:
            val = payload.get(key)
            if val and str(val).lower() in blob:
                return i
    return (category.get("digest") or [None])[0]


def match_customer_language(customer: dict[str, Any] | None) -> str:
    pref = (customer or {}).get("identity", {}).get("language_pref", "")
    if "hi-en" in pref:
        return "hi-en"
    if pref == "hi":
        return "hi"
    return "en"


# ---------------------------------------------------------------------------
# Decision selection
# ---------------------------------------------------------------------------


def trigger_category_compatible(category: dict[str, Any], trigger: dict[str, Any]) -> bool:
    slug = category.get("slug") or ""
    kind = trigger.get("kind", "")
    allowed = {
        "chronic_refill_due": {"pharmacies"},
        "supply_alert": {"pharmacies"},
        "wedding_package_followup": {"salons"},
        "cde_opportunity": {"dentists"},
        "ipl_match_today": {"restaurants"},
    }
    if kind in allowed:
        return slug in allowed[kind]
    return True


def score_trigger(category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any], customer: dict[str, Any] | None, now: str | None = None) -> float:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {}) or {}
    if not trigger_category_compatible(category, trigger):
        return -100
    score = trigger_strength(kind) + float(trigger.get("urgency", 1) or 1) * 2

    if trigger.get("scope") == "customer":
        score += 4
        if customer and customer.get("state") in {"lapsed_soft", "lapsed_hard"}:
            score += 2

    # Prefer explicit merchant intent and operationally actionable changes.
    if kind == "active_planning_intent": score += 5
    if kind in {"recall_due", "appointment_tomorrow", "chronic_refill_due", "supply_alert", "regulation_change"}: score += 3
    if kind == "curious_ask_due" and "engaged_in_last_24h" in merchant.get("signals", []): score += 3
    if kind == "dormant_with_vera" and "no_recent_conversation" in merchant.get("signals", []): score += 2

    # Specific payload = strong scoring signal. Placeholder/generated triggers get a discount.
    if payload.get("placeholder"):
        score -= 2
    else:
        concrete = sum(1 for v in payload.values() if isinstance(v, (str, int, float, list)))
        score += min(4, concrete * 0.5)

    # Do not pile onto a merchant who was just contacted successfully.
    recent = MERCHANT_SEND_MEMORY.get(merchant.get("merchant_id", ""), [])
    if recent:
        last = recent[-1]
        if kind == last.get("kind"):
            score -= 6
        elif kind in {"curious_ask_due", "research_digest", "festival_upcoming"} and (time.time() - last.get("ts", 0) < 10 * 60):
            score -= 2

    # Respect expired triggers when expires_at is parseable.
    exp = trigger.get("expires_at")
    if exp and now:
        try:
            e = datetime.fromisoformat(exp.replace("Z", "+00:00"))
            n = datetime.fromisoformat(now.replace("Z", "+00:00"))
            if e < n:
                score -= 100
        except Exception:
            pass

    # Customer send should be suppressed if consent is absent.
    if trigger.get("scope") == "customer" and not has_consent_for(customer, kind):
        score -= 100
    return score


def choose_triggers(available: list[str], now: str | None = None) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]]:
    candidates = []
    for tid in available:
        tinfo = CONTEXTS.get(("trigger", tid))
        if not tinfo:
            continue
        trigger = tinfo["payload"]
        mid = trigger.get("merchant_id")
        minfo = CONTEXTS.get(("merchant", mid))
        if not minfo:
            continue
        merchant = minfo["payload"]
        cat_slug = merchant.get("category_slug") or trigger.get("payload", {}).get("category")
        cinfo = CONTEXTS.get(("category", cat_slug))
        if not cinfo:
            continue
        category = cinfo["payload"]
        customer = None
        cid = trigger.get("customer_id")
        if cid:
            ci = CONTEXTS.get(("customer", cid))
            if ci: customer = ci["payload"]
        s = score_trigger(category, merchant, trigger, customer, now)
        if s > -50 and tid not in SENT_TRIGGER_IDS:
            candidates.append((s, trigger, category, merchant, customer))

    # One attention-preserving proactive message per merchant per tick.
    best_by_merchant: dict[str, tuple[float, dict, dict, dict, Optional[dict]]] = {}
    for item in candidates:
        s, trigger, category, merchant, customer = item
        mid = merchant.get("merchant_id")
        cur = best_by_merchant.get(mid)
        if cur is None or s > cur[0]:
            best_by_merchant[mid] = item
    chosen = sorted(best_by_merchant.values(), key=lambda x: (-x[0], x[1].get("id", "")))
    return [(t, c, m, cu) for _, t, c, m, cu in chosen]


# ---------------------------------------------------------------------------
# Deterministic message composition
# ---------------------------------------------------------------------------


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    kind = trigger.get("kind", "")
    p = trigger.get("payload", {}) or {}
    slug = merchant.get("category_slug") or category.get("slug")
    person = salutation(category, merchant, customer)
    send_as = "merchant_on_behalf" if customer else "vera"
    cta = "none"
    body = ""
    rationale = ""
    template_name = f"vera_{kind}_v1"
    template_params: list[str] = []

    if not trigger_category_compatible(category, trigger):
        # Keep the customer safe: do not send a category-incompatible operational message
        # to the customer. Instead, surface a merchant-facing data-quality hold, which is
        # useful in the static submission and is still suppressed from /v1/tick.
        holder = owner_name(merchant)
        category_label = {"dentists": "dental", "salons": "salon", "restaurants": "restaurant", "gyms": "gym", "pharmacies": "pharmacy"}.get(slug, slug.replace("_", " "))
        merchant_body = (
            f"{holder}, I’m holding this reminder because the profile is a {category_label} practice, "
            f"while the supplied reminder is pharmacy-specific. I won’t send a customer medication message "
            f"from mismatched context."
        )
        return {
            "body": merchant_body,
            "cta": "none",
            "send_as": "vera",
            "suppression_key": trigger.get("suppression_key", trigger.get("id", kind)),
            "rationale": f"Suppressed because trigger kind '{kind}' is not compatible with category '{slug}'; surfaced a merchant-facing data-quality hold instead of sending the wrong customer message.",
            "template_name": template_name,
            "template_params": [holder],
            "suppressed": True,
        }

    if customer and not has_consent_for(customer, kind):
        return {
            "body": "",
            "cta": "none",
            "send_as": send_as,
            "suppression_key": trigger.get("suppression_key", trigger.get("id", kind)),
            "rationale": "Suppressed because the available customer consent/preferences do not authorize this outreach.",
            "template_name": template_name,
            "template_params": template_params,
            "suppressed": True,
        }

    # ---------------- merchant-facing ----------------
    if not customer and kind in {"research_digest", "category_research_digest_release"}:
        item = best_digest_item(category, trigger)
        if item:
            title = item.get("title", "this week's digest item")
            source = item.get("source")
            trial_n = item.get("trial_n")
            actionable = item.get("actionable")
            extra = f" ({trial_n:,}-person study)" if isinstance(trial_n, int) else ""
            signal = ""
            if "high_risk_adult_cohort" in merchant.get("signals", []) and item.get("patient_segment") == "high_risk_adults":
                signal = " This is especially relevant to your high-risk adult cohort."
            body = f"{person}, {source or 'a new category update'} just landed: {title}{extra}.{signal}"
            if actionable:
                body += f" Practical angle: {actionable}."
            body += " Want me to turn it into a 3-line WhatsApp/GBP draft?"
            cta = "open_ended"
            rationale = "Selected a fresh, source-backed knowledge item and tied it to a merchant-specific signal before offering a concrete artifact."
        else:
            body = f"{person}, a new {slug} update landed. Want the most relevant item for your current profile?"
            cta = "open_ended"
            rationale = "No specific digest item was available, so the message avoids invented research details."

    elif not customer and kind == "regulation_change":
        item = best_digest_item(category, trigger)
        deadline = p.get("deadline_iso")
        title = item.get("title") if item else "a regulatory update"
        source = item.get("source") if item else None
        body = f"{person}, heads-up: {title}."
        if deadline and deadline[:10] not in title:
            body += f" Deadline: {deadline[:10]}."
        if item and item.get("actionable"):
            body += f" Action: {item['actionable']}."
        if source:
            body += f" — {source}"
        body += " Want me to turn that into a quick audit checklist?"
        cta = "open_ended"
        rationale = "Compliance is handled as a precise, source-cited action rather than a generic alert."

    elif not customer and kind == "perf_dip":
        metric = p.get("metric")
        delta_value = p.get("delta_pct")
        # Generated triggers can be placeholders; recover the strongest real merchant delta.
        if p.get("placeholder") or metric is None:
            d = merchant.get("performance", {}).get("delta_7d", {}) or {}
            neg = [(k, v) for k, v in d.items() if isinstance(v, (int, float)) and v < 0]
            if neg:
                metric, delta_value = min(neg, key=lambda x: x[1])
        metric = (metric or "performance").replace("_pct", "")
        delta = pct_abs(delta_value) or None
        baseline = p.get("vs_baseline")
        peer = category.get("peer_stats", {}) or {}
        peer_metric = peer.get(f"avg_{metric}_30d")
        # If a generated perf_dip trigger conflicts with the live merchant snapshot,
        # surface the contradiction instead of manufacturing a metric.
        if p.get("placeholder") and not any(isinstance(v, (int, float)) and v < 0 for v in (merchant.get("performance", {}).get("delta_7d", {}) or {}).values()):
            d7 = merchant.get("performance", {}).get("delta_7d", {}) or {}
            ups = [
                f"{k.replace('_pct', '')} are up {pct_abs(v)}" if k.replace('_pct', '') in {"calls", "views", "directions", "leads"}
                else f"{k.replace('_pct', '')} is up {pct_abs(v)}"
                for k, v in d7.items() if isinstance(v, (int, float)) and v > 0
            ]
            body = f"{person}, a performance-dip alert is active, but your current merchant snapshot is not showing a dip"
            if ups: body += f" ({', '.join(ups)} over 7d)"
            body += ". I won’t invent the missing metric. Want me to reconcile the alert before we act?"
            cta = "open_ended"
            rationale = "Detected a placeholder dip trigger that conflicts with the merchant's current positive deltas; chose verification over fabrication."
            metric = None
        else:
            body = f"{person}, {metric} are down {delta or 'in the latest window'} over the {p.get('window', '7d')}" if metric in {'calls','views','directions','leads'} else f"{person}, {metric} is down {delta or 'in the latest window'} over the {p.get('window', '7d')}"
        if metric is not None:
            current_value = (merchant.get("performance", {}) or {}).get(metric)
            if baseline is not None:
                body += f" (baseline {baseline})."
            else:
                body += "."
            if current_value is not None and peer_metric is not None:
                body += f" Current 30-day {metric} is {current_value}; category peer average is {peer_metric}."
            elif peer_metric is not None:
                body += f" Category peer average is {peer_metric}."
            sig = ", ".join(s.replace("_", " ") for s in merchant.get("signals", [])[:2])
            if sig:
                body += f" Your profile also flags {sig}."
            body += " Want me to draft the smallest fix worth testing first?"
            cta = "open_ended"
            rationale = "Anchored the dip to the exact trigger metric and benchmarked it only when a category peer value exists."

    elif not customer and kind == "perf_spike":
        metric = p.get("metric")
        delta_value = p.get("delta_pct")
        if p.get("placeholder") or metric is None:
            d = merchant.get("performance", {}).get("delta_7d", {}) or {}
            pos = [(k, v) for k, v in d.items() if isinstance(v, (int, float)) and v > 0]
            if pos:
                metric, delta_value = max(pos, key=lambda x: x[1])
        metric = (metric or "performance").replace("_pct", "")
        delta = pct_abs(delta_value) or None
        driver = p.get("likely_driver")
        if metric in {"calls", "views", "directions", "leads"}:
            body = f"{person}, good movement: {metric} are up {delta or 'in the latest window'} over {p.get('window', '7d')}"
        else:
            body = f"{person}, good movement: {metric} is up {delta or 'in the latest window'} over {p.get('window', '7d')}"
        if p.get("vs_baseline") is not None:
            body += f" vs baseline {p['vs_baseline']}"
        body += "."
        if driver:
            body += f" The trigger points to {driver.replace('_', ' ')} as the likely driver."
        body += " Want me to turn what worked into the next post/campaign draft?"
        cta = "open_ended"
        rationale = "Uses the observed improvement as a learning opportunity rather than adding a generic congratulations."

    elif not customer and kind == "seasonal_perf_dip":
        metric = p.get("metric", "views")
        delta = pct(p.get("delta_pct")) or "down"
        note = p.get("season_note")
        body = f"{person}, {metric} is down {delta} in the last {p.get('window', '7d')} — and this trigger marks it as expected seasonal movement."
        if note:
            body += f" Context: {note.replace('_', ' ')}."
        body += " I’d prioritize retention/operations over chasing acquisition this week. Want a 1-page action plan?"
        cta = "open_ended"
        rationale = "Reframes an explicitly expected seasonal dip so the merchant does not overreact."

    elif not customer and kind == "ipl_match_today":
        match = p.get("match", "today's match")
        venue = p.get("venue")
        t = p.get("match_time_iso", "")
        time_part = t[11:16] if len(t) >= 16 else ""
        digest = best_digest_item(category, {"payload": {"top_item_id": "d_2026W17_ipl_window"}})
        body = f"Quick heads-up {person} — {match}"
        if venue: body += f" at {venue}"
        if time_part: body += f" at {time_part}"
        body += " today."
        if p.get("is_weeknight") is False:
            body += " It’s a weekend match, so don’t assume the usual weeknight match-night lift."
        if digest and "12%" in json.dumps(digest):
            body += " The category digest says Saturday match covers have been 12% below Saturday average."
        offer = first_active_offer(merchant, ["bogo", "pizza", "delivery", "match"])
        if offer:
            body += f" Your active offer is {offer.get('title')}; that is the one I’d reuse for delivery rather than creating a new discount."
        body += " Want me to draft the message/banner?"
        cta = "open_ended"
        rationale = "Interprets the match-day trigger using day-of-week context and the merchant's real active offer."

    elif not customer and kind == "review_theme_emerged":
        theme = p.get("theme", "review theme").replace("_", " ")
        occ = p.get("occurrences_30d")
        quote = p.get("common_quote")
        body = f"{person}, one review theme is rising: {theme}"
        if occ is not None: body += f" ({occ} mentions in 30 days)"
        body += "."
        if quote: body += f" One example: \"{quote}\"."
        body += " This is specific enough to test an operational fix. Want me to draft a response + fix checklist?"
        cta = "open_ended"
        rationale = "Uses repeated review evidence and converts it into a concrete merchant action."

    elif not customer and kind == "milestone_reached":
        metric = p.get("metric")
        cur = p.get("value_now")
        target = p.get("milestone_value")
        if metric is not None and cur is not None:
            body = f"{person}, you’re at {cur} {str(metric).replace('_', ' ')}"
            if target is not None:
                try:
                    remaining = max(0, int(target) - int(cur))
                    body += f" — only {remaining} to reach {target}."
                except Exception:
                    body += f" with the milestone target at {target}."
            else:
                body += "."
            body += " Want a simple way to turn the milestone into the next customer-facing post?"
            rationale = "Makes the milestone concrete and offers a low-effort use of it."
        else:
            body = f"{person}, a milestone is flagged for your profile, but the current event payload does not include the exact metric. I won’t guess. Want me to prepare a milestone post once the number lands?"
            rationale = "The generated trigger omits the milestone metric, so the message preserves trust instead of fabricating a number."
        cta = "open_ended"

    elif not customer and kind == "competitor_opened":
        name = p.get("competitor_name")
        dist = p.get("distance_km")
        comp_offer = p.get("their_offer")
        if name:
            body = f"{person}, {name} opened nearby"
            if dist is not None: body += f" ({dist} km)"
            body += "."
            if comp_offer: body += f" Their listed offer is {comp_offer}."
            body += " I’d use this as a positioning check, not a price war. Want me to compare their visible offer with your current profile?"
        else:
            body = f"{person}, a nearby-competitor trigger is active, but the current event payload doesn’t include the competitor details. I won’t invent them. Want a positioning checklist using your current offer/profile?"
        cta = "open_ended"
        rationale = "Reports the local competitive event factually and avoids unverified claims about the competitor."

    elif not customer and kind == "renewal_due":
        days = p.get("days_remaining", merchant.get("subscription", {}).get("days_remaining"))
        amount = money(p.get("renewal_amount"))
        plan = p.get("plan", merchant.get("subscription", {}).get("plan"))
        body = f"{person}, your {plan or 'subscription'} renews in {days} days"
        if amount: body += f" at {amount}"
        body += "."
        body += " Before you renew, want a quick summary of what’s working vs what I’d change?"
        cta = "open_ended"
        rationale = "Uses the exact renewal horizon and frames the message around value review rather than pressure."

    elif not customer and kind == "gbp_unverified":
        path = p.get("verification_path", "the available verification path")
        body = f"{person}, your Google Business Profile is still unverified."
        body += f" The context shows verification via {path}."
        if p.get("estimated_uplift_pct") is not None:
            body += f" The trigger estimates up to {pct(p['estimated_uplift_pct'])} upside, but I’d treat that as an estimate, not a guarantee."
        body += " Want a step-by-step checklist for the verification flow?"
        cta = "open_ended"
        rationale = "Keeps the verification alert factual while clearly labeling the supplied uplift as an estimate."

    elif not customer and kind == "supply_alert":
        molecule = p.get("molecule", "the affected medicine")
        batches = ", ".join(p.get("affected_batches", []) or [])
        manufacturer = p.get("manufacturer")
        body = f"{person}, supply alert for {molecule}"
        if batches: body += f" — affected batches: {batches}"
        if manufacturer: body += f" ({manufacturer})"
        body += ". I’d isolate/verify those batches against your stock before any dispensing decision. Want a batch-audit checklist?"
        cta = "open_ended"
        rationale = "Treats a product alert as a precise operational check and avoids giving patient-medication advice."

    elif not customer and kind == "category_seasonal":
        trends = p.get("trends") or []
        body = f"{person}, the current category shift is clear: " + ", ".join(str(x).replace("_", " ") for x in trends[:3]) + "."
        body += ""
        if p.get("shelf_action_recommended"):
            body += " This trigger recommends a shelf/display adjustment."
        body += " Want me to turn the trend into a simple shelf + WhatsApp action list?"
        cta = "open_ended"
        rationale = "Uses the trigger’s exact demand shifts and turns them into a category-appropriate operational action."

    elif not customer and kind == "festival_upcoming":
        festival = p.get("festival")
        days = p.get("days_until")
        date = p.get("date")
        if festival:
            body = f"{person}, {festival} is coming up"
            if date: body += f" on {date}"
            elif days is not None: body += f" in {days} days"
            body += ". I’d plan the category-specific offer/content now rather than send a generic discount. Want me to draft the first version?"
            rationale = "Uses the supplied festival timing and keeps the recommendation category-led."
        else:
            body = f"{person}, a festival-upcoming trigger is active, but its exact event details haven’t been supplied yet. I won’t guess. Want a reusable seasonal planning checklist for {merchant.get('identity', {}).get('locality', 'your area')}?"
            rationale = "The trigger is underspecified, so the message avoids inventing a festival name or date."
        cta = "open_ended"

    elif not customer and kind == "cde_opportunity":
        item = find_by_id(category.get("digest", []) or [], p.get("digest_item_id"))
        title = item.get("title") if item else "a relevant continuing-education session"
        when = item.get("date") if item else None
        body = f"{person}, {title}"
        if when: body += f" is on {when[:16].replace('T', ' ')}"
        body += "."
        if p.get("credits") is not None: body += f" {p['credits']} CDE credits"
        if p.get("fee"): body += f"; {p['fee'].replace('_', ' ')}."
        else: body += "."
        body += " Want me to pull the practical takeaways before you decide?"
        cta = "open_ended"
        rationale = "Turns a professional-development trigger into a low-friction relevance check."

    elif not customer and kind == "curious_ask_due":
        # Ask from actual category trend data when available, not a fabricated guess.
        trend = (category.get("trend_signals") or [None])[0]
        topic = trend.get("query") if trend else None
        if topic:
            body = f"{person}, quick one — have you noticed more people asking about \"{topic}\" lately?"
        else:
            body = f"{person}, quick one — what service/topic has been getting the most customer questions this week?"
        body += " I’ll turn your answer into a short GBP post + reply you can reuse."
        cta = "open_ended"
        rationale = "Uses curiosity and reciprocity: one easy question, followed by a concrete artifact."

    elif not customer and kind == "dormant_with_vera":
        days = p.get("days_since_last_merchant_message")
        last_topic = (p.get("last_topic") or "").replace("_", " ")
        if days is not None:
            body = f"{person}, it’s been {days} days since our last conversation."
        else:
            body = f"{person}, a re-engagement window is active for your account."
        if last_topic: body += f" Last time we were on {last_topic}; I don’t want to repeat that."
        body += " I have one fresh, merchant-specific idea for your current profile. Want the 30-second version?"
        cta = "open_ended"
        rationale = "Acknowledges the gap, explicitly avoids repeating the previous topic, and asks for permission to re-engage."

    elif not customer and kind == "winback_eligible":
        days = p.get("days_since_expiry")
        dip = pct(p.get("perf_dip_pct"))
        added = p.get("lapsed_customers_added_since_expiry")
        body = f"{person}, since expiry it’s been {days} days"
        if dip: body += f" and the trigger shows a {dip} performance dip"
        body += "."
        if added is not None: body += f" {added} customers have also entered the lapsed pool since then."
        body += " Want a reactivation plan built around the current profile instead of another generic renewal pitch?"
        cta = "open_ended"
        rationale = "Combines the supplied post-expiry performance and customer signals into a single reactivation decision."

    elif not customer and kind == "active_planning_intent":
        last_msg = p.get("merchant_last_message") or ""
        history_text = " ".join(h.get("body", "") for h in merchant.get("conversation_history", [])[-4:] if h.get("from") in {"merchant", "vera"})
        combined = f"{last_msg} {history_text}".lower()
        body = f"{person}, yes — here’s a starter shape based on what you asked: "
        if "corporate_bulk" in (p.get("intent_topic", "") or "") or "corporate" in combined:
            offer = first_active_offer(merchant, ["thali", "lunch"])
            anchor = offer.get("title") if offer else "your current lunch offer"
            body += f"keep {anchor} as the retail anchor, then define 10/25/50-order bulk tiers, delivery cutoff, and billing terms with you before pricing them."
        elif "kids yoga" in combined:
            body += "4-week program, 3 classes/week, age 7–12, matching the structure already discussed in your conversation history."
        else:
            body += "one clear offer, one target customer, and one next action — using only the details already in your conversation."
        body += " Want me to draft the customer-facing copy next?"
        cta = "open_ended"
        rationale = "Treats an explicit planning intent as an action request and reuses only facts already present in merchant history."

    else:
        # ---------------- customer-facing ----------------
        if kind in {"recall_due", "appointment_tomorrow"}:
            ci = customer or {}
            cname = ci.get("identity", {}).get("name", "there")
            due = p.get("due_date")
            last = p.get("last_service_date")
            slots = p.get("available_slots") or p.get("next_session_options") or []
            offer = customer_safe_offer(merchant, customer)
            body = f"Hi {cname}, {merchant_name(merchant)} here."
            if kind == "recall_due":
                if last: body += f" Your last visit was {last}."
                if due: body += f" Your recall window is due {due}."
            else:
                body += " Your appointment is tomorrow."
            if slots:
                labels = [x.get("label") for x in slots[:2] if x.get("label")]
                if labels: body += " Available: " + " or ".join(labels) + "."
            if offer and kind == "recall_due" and not p.get("placeholder"):
                body += f" {offer.get('title')}."
            if kind == "appointment_tomorrow" and not slots:
                body += " Want me to help confirm it?"
            elif kind == "recall_due" and p.get("placeholder") and not slots:
                body += " I don’t have a confirmed slot in this update. Want me to help pick a suitable time?"
            elif match_customer_language(customer) == "hi-en":
                body += " Aapka preferred time ho to bata dijiye."
            else:
                body += " Want me to help confirm the next step?"
            cta = "open_ended"
            rationale = "Customer-facing reminder uses consent, relationship history, concrete timing and the merchant's actual active offer."

        elif kind in {"chronic_refill_due"}:
            ci = customer or {}
            cname = ci.get("identity", {}).get("name", "there")
            meds = ", ".join(p.get("molecule_list", [])[:3])
            runout = p.get("stock_runs_out_iso", "")
            body = f"Hi {cname}, {merchant_name(merchant)} here. Your regular refill reminder is due."
            if meds: body += f" The medicines on your last refill list include {meds}."
            if runout: body += f" The trigger shows stock running out around {runout[:10]}."
            if p.get("delivery_address_saved"):
                delivery = first_active_offer(merchant, ["delivery"])
                if delivery: body += f" We also have {delivery.get('title')}."
                body += " Reply if you'd like the pharmacy to prepare the refill/delivery check."
            else:
                body += " Reply if you'd like the pharmacy to confirm the refill."
            cta = "open_ended"
            rationale = "Keeps the message administrative and grounded: existing refill list, timing and saved delivery preference only."

        elif kind in {"trial_followup", "wedding_package_followup", "customer_lapsed_soft", "customer_lapsed_hard"}:
            ci = customer or {}
            cname = ci.get("identity", {}).get("name", "there")
            days = p.get("days_since_last_visit")
            offer = customer_safe_offer(merchant, customer)
            if kind == "trial_followup":
                body = f"Hi {cname}, {merchant_name(merchant)} here. Quick follow-up on your {p.get('trial_date', 'recent')} trial."
                opts = p.get("next_session_options") or []
                if opts and opts[0].get("label"): body += f" Next option: {opts[0]['label']}."
                body += " Want me to hold it?"
            elif kind == "wedding_package_followup":
                wedding = p.get("wedding_date")
                d = p.get("days_to_wedding")
                body = f"Hi {cname}, {merchant_name(merchant)} here."
                if wedding: body += f" Your wedding date is {wedding}."
                if d: body += f" That’s {d} days away."
                body += " You’re in the next-step window from your bridal trial. Want to review the options?"
            else:
                body = f"Hi {cname}, {merchant_name(merchant)} here."
                if days is not None: body += f" It’s been {days} days since your last visit."
                else:
                    last_visit = (ci.get("relationship", {}) or {}).get("last_visit")
                    if last_visit: body += f" Your last recorded visit was {last_visit}."
                focus = p.get("previous_focus")
                if focus: body += f" Your previous focus was {focus.replace('_', ' ')}."
                body += " No pressure — I just wanted to check whether you’d like to come back."
                if offer: body += f" {offer.get('title')} is currently active."
                body += " Want me to share a simple next option?"
            cta = "open_ended"
            rationale = "Win-back/follow-up copy is warm and low-pressure, using only the customer relationship state and active merchant offer."

        else:
            # Generic but grounded customer-facing fallback.
            cname = (customer or {}).get("identity", {}).get("name", "there")
            body = f"Hi {cname}, {merchant_name(merchant)} here."
            if p:
                # Use one safe trigger fact, but never expose generator placeholders.
                items = [f"{k.replace('_', ' ')}: {v}" for k, v in p.items() if isinstance(v, (str, int, float)) and k != "placeholder"]
                if items: body += " Quick update — " + items[0] + "."
            body += " Reply here if you want us to help with the next step."
            cta = "open_ended"
            rationale = "Generic fallback uses only a single supplied trigger fact to avoid hallucination."

    body = normalize_and_constrain(body)
    rationale = rationale[:500]
    suppression = trigger.get("suppression_key") or trigger.get("id") or f"{kind}:{merchant.get('merchant_id')}"
    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression,
        "rationale": rationale,
        "template_name": template_name,
        "template_params": [person] + ([customer.get("identity", {}).get("name")] if customer else []),
        "trigger_id": trigger.get("id"),
        "merchant_id": merchant.get("merchant_id"),
        "customer_id": customer.get("customer_id") if customer else None,
    }


def normalize_and_constrain(body: str) -> str:
    body = re.sub(r"\s+", " ", body or "").strip()
    # Don't let our deterministic renderer expose implementation jargon.
    body = re.sub(r"\btrigger context\b", "update", body, flags=re.I)
    return body[:1200]


# ---------------------------------------------------------------------------
# Optional LLM polish (language only; facts guarded after generation)
# ---------------------------------------------------------------------------


def polish_with_llm(draft: dict[str, Any], category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any], customer: dict[str, Any] | None) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return draft
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    evidence = sorted(evidence_values(category, merchant, trigger, customer))
    system = (
        "You are a WhatsApp copy editor for a merchant assistant. Rewrite the draft for clarity and natural Indian business tone. "
        "Do not add, remove, or alter factual details. Do not invent names, prices, dates, percentages, offers, competitors, or outcomes. "
        "Keep one clear CTA. Return only the message body, no quotes and no commentary."
    )
    prompt = json.dumps({
        "category": category.get("slug"),
        "languages": merchant.get("identity", {}).get("languages", []),
        "customer_language": (customer or {}).get("identity", {}).get("language_pref"),
        "draft": draft.get("body", ""),
        "allowed_evidence": evidence,
    }, ensure_ascii=False)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 300,
    }
    req = urlrequest.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        if validate_polished_text(text, draft, category, merchant, trigger, customer):
            polished = dict(draft)
            polished["body"] = normalize_and_constrain(text)
            polished["rationale"] = draft.get("rationale", "") + " LLM polish passed the evidence guard."
            return polished
    except Exception:
        pass
    return draft


def validate_polished_text(text: str, draft: dict[str, Any], category: dict[str, Any], merchant: dict[str, Any], trigger: dict[str, Any], customer: dict[str, Any] | None) -> bool:
    if not text or len(text) < 20 or len(text) > 900:
        return False
    low = text.lower()
    if any(j in low for j in INTERNAL_JARGON):
        return False
    taboos = set(x.lower() for x in (category_voice(category).get("vocab_taboo") or []))
    if any(t in low for t in taboos):
        return False
    allowed_nums = evidence_number_tokens(category, merchant, trigger, customer)
    nums = extract_numbers(text)
    if not nums.issubset(allowed_nums):
        return False
    # Keep the key person anchor.
    expected = (customer or {}).get("identity", {}).get("name") if customer else owner_name(merchant).replace("Dr. ", "")
    if expected and expected.lower() not in low:
        return False
    return True


# ---------------------------------------------------------------------------
# Conversation intelligence
# ---------------------------------------------------------------------------

STOP_PATTERNS = [
    r"\bstop\b", r"\bunsubscribe\b", r"\bdo not message\b", r"\bdon't message\b",
    r"\bnot interested\b", r"\bno more\b", r"\bremove me\b", r"\bspam\b",
]
COMMITMENT_PATTERNS = [
    r"\bok\b.*\b(let'?s|do it|proceed|go ahead)\b", r"\byes\b", r"\bgo ahead\b",
    r"\blet'?s do it\b", r"\bdo it\b", r"\bsend it\b", r"\bplease send\b",
    r"\bconfirm\b", r"\bbook it\b", r"\bstart it\b", r"\bi want to join\b",
]
AUTO_REPLY_PATTERNS = [
    "thank you for contacting us",
    "our team will respond shortly",
    "we will get back to you",
    "thanks for your message",
    "please wait for our team",
]
HOSTILE_WORDS = {"useless", "idiot", "stupid", "shut up", "spam"}


def classify_reply(message: str) -> str:
    low = norm(message)
    if any(re.search(p, low) for p in STOP_PATTERNS):
        return "stop"
    if any(w in low for w in HOSTILE_WORDS):
        return "hostile"
    if any(re.search(p, low) for p in COMMITMENT_PATTERNS):
        return "commitment"
    if any(p in low for p in AUTO_REPLY_PATTERNS):
        return "auto_reply"
    if any(x in low for x in ["gst", "tax filing", "income tax", "gst return"]):
        return "off_topic"
    if any(x in low for x in ["not now", "later", "busy", "tomorrow", "give me time"]):
        return "defer"
    if "?" in message or any(q in low for q in ["how", "what", "which", "why", "can you", "could you"]):
        return "question"
    return "acknowledgement"


def is_global_auto_reply(merchant_id: str, message: str) -> tuple[bool, int]:
    fp = hashlib.sha1(norm(message).encode()).hexdigest()
    hist = MERCHANT_REPLY_HISTORY[merchant_id]
    now = time.time()
    hist.append((now, fp))
    # Count repeated identical fingerprint within a 10-minute local window.
    count = sum(1 for ts, f in hist if f == fp and now - ts < 600)
    return count >= 3, count


def conversation_context(conv_id: str, merchant_id: str) -> dict[str, Any]:
    state = CONVERSATIONS.get(conv_id)
    if state:
        return state
    m = CONTEXTS.get(("merchant", merchant_id), {}).get("payload", {})
    cat_slug = m.get("category_slug")
    state = {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": None,
        "category_slug": cat_slug,
        "turns": [],
        "mode": "initial",
        "objective": "merchant_assistance",
        "last_trigger_kind": None,
        "ended": False,
    }
    CONVERSATIONS[conv_id] = state
    return state


def respond(conv_id: str, merchant_id: str, message: str, turn_number: int = 1) -> dict[str, Any]:
    state = conversation_context(conv_id, merchant_id)
    if state.get("ended"):
        return {"action": "end", "rationale": "Conversation was already closed after a stop/end signal."}

    kind = classify_reply(message)
    state["turns"].append({"role": "merchant", "body": message, "kind": kind, "turn": turn_number, "ts": now_iso()})

    auto, count = is_global_auto_reply(merchant_id, message)
    if kind == "auto_reply" and count >= 4:
        state["ended"] = True
        return {"action": "end", "rationale": f"Detected repeated canned auto-reply pattern across merchant replies (count={count}); closing rather than looping."}
    if kind == "auto_reply" and count == 1:
        return {
            "action": "send",
            "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' if this is still useful.",
            "cta": "binary_yes_no",
            "rationale": "Detected a likely canned auto-reply; one explicit owner-facing prompt is enough before backing off.",
        }
    if kind == "auto_reply":
        wait = min(900, 300 + max(0, count - 2) * 180)
        return {"action": "wait", "wait_seconds": wait, "rationale": "Repeated canned auto-reply; backing off rather than consuming another turn."}

    if kind == "stop":
        state["ended"] = True
        return {"action": "end", "rationale": "Merchant explicitly asked Vera to stop or declined further contact."}

    if kind == "hostile":
        return {
            "action": "send",
            "body": "Understood — sorry for the interruption. I’ll keep this thread closed unless you choose to come back to it.",
            "cta": "none",
            "rationale": "Acknowledged the hostile/stop signal without arguing or continuing the sales flow.",
        }

    if kind == "commitment":
        state["mode"] = "action"
        state["objective"] = state.get("objective") or "merchant_assistance"
        obj = state.get("last_trigger_kind") or state.get("objective") or "merchant_assistance"
        # Crucially: do not re-qualify. Continue with a concrete next step tied to the thread.
        if obj in {"research_digest", "category_research_digest_release", "cde_opportunity"}:
            body = "Great — I’ll turn the research item already shared into the practical takeaway and draft from this thread. Reply CONFIRM when you want the draft finalized."
            cta = "binary_confirm_cancel"
        elif obj == "active_planning_intent":
            body = "Great — I’ll turn the agreed outline into the customer-facing draft now, using the details already in this thread. Reply CONFIRM when you want to use it."
            cta = "binary_confirm_cancel"
        elif obj in {"perf_dip", "seasonal_perf_dip", "perf_spike"}:
            body = "Great — I’ll prepare the smallest testable change from the performance signal already discussed. Reply CONFIRM when you want the draft."
            cta = "binary_confirm_cancel"
        elif state.get("customer_id"):
            body = "Done — I’ll move to the next step now, using the customer and offer details already shared rather than asking you to repeat them."
            cta = "open_ended"
        else:
            body = "Done — I’ll move to the next step using the details already in this thread. Reply with any edit you want me to make."
            cta = "open_ended"
        return {
            "action": "send",
            "body": body,
            "cta": cta,
            "rationale": "Explicit commitment detected; switched from qualification to execution with a thread-specific next step.",
        }

    if kind == "defer":
        state["mode"] = "wait"
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Merchant asked for time; pausing rather than pushing another question."}

    if kind == "question":
        state["mode"] = "qualifying"
        # Use the most recently sent message as the local object of the question.
        return {
            "action": "send",
            "body": "Yes — I can help with that. I’ll keep the answer tied to the merchant data already in this thread; what part should I answer first?",
            "cta": "open_ended",
            "rationale": "The merchant asked a genuine question, so the assistant stays in clarification mode without opening a new sales pitch.",
        }

    if kind == "off_topic":
        return {
            "action": "send",
            "body": "I can help with the current merchant task in this chat, but GST/tax filing is outside this Vera flow. I’ll stay on the business action we were working on.",
            "cta": "none",
            "rationale": "Stayed on-mission instead of inventing an unrelated capability.",
        }

    state["mode"] = "engaged"
    return {
        "action": "send",
        "body": "Got it. I’ll build on the details you’ve already shared rather than restarting the qualification.",
        "cta": "open_ended",
        "rationale": "Acknowledged the merchant and preserved conversation continuity.",
    }


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


@app.get("/v1/healthz")
async def healthz() -> dict[str, Any]:
    counts = defaultdict(int)
    for scope, _ in CONTEXTS:
        counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 3),
        "contexts_loaded": {
            "category": counts["category"],
            "merchant": counts["merchant"],
            "customer": counts["customer"],
            "trigger": counts["trigger"],
        },
    }


@app.get("/v1/metadata")
async def metadata() -> dict[str, Any]:
    return {
        "team_name": os.getenv("TEAM_NAME", "Vera Signal-to-Action"),
        "team_members": [x.strip() for x in os.getenv("TEAM_MEMBERS", "Sumit").split(",") if x.strip()],
        "model": os.getenv("OPENAI_MODEL", "deterministic-with-optional-llm-polish"),
        "approach": "signal ranking + evidence guard + attention budget + conversation state", 
        "contact_email": os.getenv("CONTACT_EMAIL", ""),
        "version": "2.0.0",
        "submitted_at": os.getenv("SUBMITTED_AT", now_iso()),
    }


@app.post("/v1/context")
async def push_context(request: Request) -> dict[str, Any]:
    data = await request.json()
    scope = data.get("scope")
    cid = data.get("context_id")
    version = data.get("version")
    payload = data.get("payload")
    if scope not in {"category", "merchant", "customer", "trigger"}:
        return {"accepted": False, "reason": "invalid_scope", "details": "scope must be category|merchant|customer|trigger"}
    if not cid or not isinstance(version, int) or payload is None:
        return {"accepted": False, "reason": "invalid_payload", "details": "context_id, integer version and payload are required"}
    key = (scope, cid)
    current = CONTEXTS.get(key)
    if current and version < current["version"]:
        return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}
    if current and version == current["version"]:
        return {"accepted": True, "ack_id": f"ack_{scope}_{cid}_v{version}", "stored_at": now_iso()}
    CONTEXTS[key] = {"version": version, "payload": payload, "updated_at": now_iso()}
    return {"accepted": True, "ack_id": f"ack_{scope}_{cid}_v{version}", "stored_at": now_iso()}


@app.post("/v1/tick")
async def tick(request: Request) -> dict[str, Any]:
    data = await request.json()
    now = data.get("now") or now_iso()
    available = data.get("available_triggers") or []
    chosen = choose_triggers(available, now)
    actions = []
    for trigger, category, merchant, customer in chosen[:20]:
        # Suppression check against trigger-level key and merchant attention budget.
        key = trigger.get("suppression_key") or trigger.get("id")
        mid = merchant.get("merchant_id")
        recent = MERCHANT_SEND_MEMORY.get(mid, [])
        if any(x.get("suppression_key") == key for x in recent):
            continue

        draft = compose(category, merchant, trigger, customer)
        if draft.get("suppressed") or not draft.get("body"):
            continue
        # Store conversation state for replies.
        conv_id = f"conv_{mid}_{trigger.get('id')}_{len(recent)+1}"
        CONVERSATIONS[conv_id] = {
            "conversation_id": conv_id,
            "merchant_id": mid,
            "customer_id": customer.get("customer_id") if customer else None,
            "category_slug": merchant.get("category_slug"),
            "turns": [{"role": "vera", "body": draft["body"], "ts": now_iso()}],
            "mode": "outreach",
            "objective": trigger.get("kind"),
            "last_trigger_kind": trigger.get("kind"),
            "ended": False,
        }
        if os.getenv("ENABLE_LLM_POLISH", "0") == "1":
            draft = polish_with_llm(draft, category, merchant, trigger, customer)
        action = {
            "conversation_id": conv_id,
            "merchant_id": mid,
            "customer_id": customer.get("customer_id") if customer else None,
            "send_as": draft["send_as"],
            "trigger_id": trigger.get("id"),
            "template_name": draft.get("template_name"),
            "template_params": draft.get("template_params", []),
            "body": draft["body"],
            "cta": draft["cta"],
            "suppression_key": draft["suppression_key"],
            "rationale": draft["rationale"],
        }
        actions.append(action)
        SENT_TRIGGER_IDS.add(trigger.get("id"))
        MERCHANT_SEND_MEMORY[mid].append({"ts": time.time(), "kind": trigger.get("kind"), "suppression_key": key, "trigger_id": trigger.get("id")})

    return {"actions": actions[:20]}


@app.post("/v1/reply")
async def reply(request: Request) -> dict[str, Any]:
    data = await request.json()
    conv_id = data.get("conversation_id", "conv_unknown")
    merchant_id = data.get("merchant_id", "")
    customer_id = data.get("customer_id")
    message = data.get("message", "")
    turn_number = int(data.get("turn_number", 1) or 1)
    state = conversation_context(conv_id, merchant_id)
    if customer_id and not state.get("customer_id"):
        state["customer_id"] = customer_id
    result = respond(conv_id, merchant_id, message, turn_number)
    if result.get("action") == "send":
        state["turns"].append({"role": "vera", "body": result.get("body", ""), "ts": now_iso()})
    return result


# ---------------------------------------------------------------------------
# Local function for direct canonical composition tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)
