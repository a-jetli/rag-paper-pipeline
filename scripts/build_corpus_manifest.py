"""
Phase 1: Corpus selection.

Picks ~1000 papers (hand-picked anchors + per-topic keyword fill to quota),
fetches their metadata from arXiv, and writes data/corpus_manifest.json.

This does NOT download PDFs or build embeddings. That is Phase 2 (build_index.py),
which consumes the manifest. Every arXiv response is cached to cache/arxiv/ so
re-running while tuning filters is instant.
"""

import os
import re
import sys
import json
import time
import hashlib
import requests
import feedparser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT = 3
CACHE_DIR = "cache/arxiv"
MANIFEST_PATH = "data/corpus_manifest.json"

FIELD_QUOTAS = {
    "retrieval_rag": 130, "llms_scaling": 90, "transformers_architecture": 70,
    "generative_models": 70, "classic_deep_learning": 59, "computer_vision": 59,
    "training_alignment": 60, "efficiency": 55, "embeddings": 50,
    "prompting_reasoning": 50, "reinforcement_learning": 49, "agents_tools": 40,
    "multimodal": 40, "evaluation": 36, "hallucination_factuality": 34,
    "interpretability_safety": 29, "long_context": 21, "speech_audio": 21,
    "classical_ml": 21, "graph_neural_networks": 16,
}  # sums to 1000 (+7 surveys, -7 dead-link/unparseable fill papers)

FIELD_KEYWORDS = {
    "retrieval_rag": "retrieval augmented generation dense passage retrieval",
    "llms_scaling": "large language models scaling laws",
    "transformers_architecture": "transformer self-attention language model",
    "generative_models": "diffusion models generative adversarial",
    "classic_deep_learning": "deep convolutional neural network image",
    "computer_vision": "computer vision object detection segmentation",
    "training_alignment": "alignment RLHF instruction tuning preference",
    "efficiency": "efficient inference quantization parameter LLM",
    "embeddings": "text embeddings sentence representation learning",
    "prompting_reasoning": "chain of thought prompting reasoning language models",
    "reinforcement_learning": "deep reinforcement learning policy",
    "agents_tools": "language model agents tool use planning",
    "multimodal": "multimodal vision language model",
    "evaluation": "evaluation benchmark language models",
    "hallucination_factuality": "hallucination factuality language models",
    "interpretability_safety": "interpretability explainability mechanistic language model",
    "long_context": "long context window language models",
    "speech_audio": "speech recognition audio self-supervised",
    "classical_ml": "gradient boosting decision trees random forest support vector machine",
    "graph_neural_networks": "graph neural network graph representation learning",
}

# Quality gates applied to keyword-fill candidates (NOT to hand-picks).
ALLOWED_CAT_PREFIXES = ("cs.", "stat.ML", "eess.")
BLOCKED_PRIMARY = {"cs.AR", "cs.DC", "cs.NI", "cs.OS", "cs.PF"}  # hardware/systems/networking
MAX_FILL_YEAR = "2025"  # drop fill papers newer than this (too fresh to be canonical)

# Specific fill papers judged off-topic on review; excluded so the next-best candidate backfills.
FILL_EXCLUDE = {
    "1908.07590", "2504.13684", "0903.2544", "1606.07660",          # retrieval strays
    "2502.09173", "2510.21894",                                      # llms_scaling strays
    "2510.17851", "1910.08108", "2311.06480", "2005.11626",         # generative strays
    "2311.04552", "2303.09295", "2311.03226", "1905.00307",
    "1803.02421", "2107.02926", "1905.00322",                       # classic_dl strays
    "2311.16338", "2511.21779",                                     # alignment strays
    "2505.18332", "2104.06667",                                     # efficiency strays
    "1812.01662", "2011.01014", "2302.08893", "1812.09225",         # embeddings strays
    "1905.12588", "2510.01658",
    "2504.16021",                                                    # prompting stray
    "1610.00031", "1503.07220", "2306.07353", "2008.00969",         # agents strays
    "2110.02480", "1610.00030", "2011.01774",
    "2008.13369",                                                    # multimodal stray
    "1906.07008", "2512.03107",                                     # hallucination strays
    "cmp-lg/9701001",                                               # long_context stray
    "2402.00054",                                                    # classical_ml stray (genetics)
    # dropped after ingestion: 4 dead arXiv links (404) + 3 unparseable PDFs
    "1807.01418", "1310.0319", "2107.10998", "1804.06309",
    "1906.05651", "2401.06855", "2510.03799",
}
TITLE_BLOCKLIST = [
    # science/medicine
    "clinical", "medical", "healthcare", "electronic health", "biomedical",
    "biology", "molecular", "protein", "genomic", "chemistry", "drug",
    "disease", "tumor", "diagnos", "patient", "eeg", "ecg", "cancer",
    "covid", "galax", "astronom", "cosmic", "physics-informed", "seismic",
    # agriculture / environment
    "agricultur", "tomato", "crop", "wheat", "soil", "remote sensing",
    "satellite", "weather", "climate", "earthquake", "flood", "wildfire",
    # finance / business / security
    "financial", "stock market", "fraud", "credit risk", "loan", "insurance",
    "churn", "e-commerce", "advertis", "manufacturing", "supply chain",
    "cryptanalysis", "malware", "intrusion", "phishing", "blockchain",
    "cryptocurr", "iot", "smart grid", "energy consumption",
    # telecom / robotics / transport
    "wireless", "5g", "6g", "autonomous driving", "vehicle", "pedestrian",
    "drone", "uav", "robot",
    # arts / social / games (applied niches, not foundational)
    "music", "song", "lyric", "poetry", "sentiment", "fake news",
    "hate speech", "twitter", "social media", "recommend", "dialogue",
    "community question", "sport", "soccer", "basketball", "poker",
    "video game", "game level",
]


def passes_filters(paper: dict) -> bool:
    """Keep a fill candidate only if it's in a CS/ML category, recent-but-not-bleeding-edge,
    and not a hyper-applied niche paper."""
    cats = paper.get("categories", [])
    if cats and cats[0] in BLOCKED_PRIMARY:
        return False
    if not any(c.startswith(ALLOWED_CAT_PREFIXES) for c in cats):
        return False
    year = (paper.get("published") or "")[:4]
    if year and year > MAX_FILL_YEAR:
        return False
    title_l = paper.get("title", "").lower()
    if any(term in title_l for term in TITLE_BLOCKLIST):
        return False
    return True

# Hand-picked anchors (verified against arXiv: 140/140 resolve correctly).
ANCHORS = {
    "transformers_architecture": [
        "1706.03762", "1810.04805", "1910.10683", "2005.14165", "1907.11692",
        "2003.10555", "1901.02860", "1909.11942", "1607.06450", "2104.09864",
        "2312.00752", "2401.04088", "2101.03961",
    ],
    "llms_scaling": [
        "2001.08361", "2203.15556", "2206.07682", "2204.02311", "2302.13971",
        "2307.09288", "2407.21783", "2303.08774", "2310.06825", "2412.19437",
        "2501.12948", "2407.10671", "2404.14219",
    ],
    "retrieval_rag": [
        "2005.11401", "2004.04906", "2004.12832", "2002.08909", "2007.01282",
        "2208.03299", "2310.11511", "2212.10496", "2112.04426", "2307.03172",
        "2312.10997", "2401.18059", "2401.15884", "2404.16130",
    ],
    "embeddings": [
        "1301.3781", "1908.10084", "2104.08821", "2212.03533", "2112.09118",
        "2205.13147", "1802.05365",
    ],
    "training_alignment": [
        "2203.02155", "2305.18290", "2212.08073", "2305.11206", "2109.01652",
        "2212.10560", "1706.03741", "1707.06347",
    ],
    "prompting_reasoning": [
        "2201.11903", "2203.11171", "2305.10601", "2210.03629", "2205.10625",
        "2205.11916", "2203.14465", "2305.20050",
    ],
    "hallucination_factuality": [
        "2202.03629", "2303.08896", "2305.14251", "2109.07958",
    ],
    "efficiency": [
        "2106.09685", "2305.14314", "2205.14135", "2307.08691", "2210.17323",
        "1503.02531", "1910.01108", "2306.00978", "1909.08053", "1910.02054",
    ],
    "evaluation": [
        "2009.03300", "2211.09110", "2206.04615", "2306.05685", "2309.15217",
        "1606.05250", "1804.07461",
    ],
    "classic_deep_learning": [
        "1512.03385", "1502.03167", "1412.6980", "1409.1556", "1409.3215",
        "1409.0473", "1505.04597", "1409.4842",
    ],
    "computer_vision": [
        "2010.11929", "2103.00020", "1703.06870", "1506.01497", "1506.02640",
        "2304.02643", "2104.14294", "1905.11946",
    ],
    "generative_models": [
        "1406.2661", "1312.6114", "2006.11239", "2011.13456", "2112.10752",
        "2102.12092", "1812.04948", "1703.10593", "2207.12598", "2303.01469",
    ],
    "reinforcement_learning": [
        "1312.5602", "1712.01815", "1911.08265", "1801.01290", "2106.01345",
    ],
    "agents_tools": [
        "2302.04761", "2305.16291", "2303.11366", "2310.06770", "2307.13854",
        "2304.03442",
    ],
    "multimodal": [
        "2204.14198", "2301.12597", "2304.08485", "2403.05530",
    ],
    "speech_audio": [
        "2212.04356", "2006.11477",
    ],
    "long_context": [
        "2004.05150", "2305.13048", "2310.01889", "2306.15595",
    ],
    "interpretability_safety": [
        "1606.06565", "2202.05262", "2303.12712", "2307.15043",
    ],
    "graph_neural_networks": [
        "1609.02907", "1706.02216", "1710.10903",
    ],
    "classical_ml": [
        "1603.02754", "1201.0490",
    ],
}

# Additional hand-picked canonical papers (resolved by title-search + ID recovery,
# all verified to match their arXiv title). Folded in as guaranteed slots alongside ANCHORS.
CURATED = {
    "retrieval_rag": ["2112.01488", "2007.00808", "2010.08191", "2301.12652", "2212.10509",
                      "1911.00172", "2109.10086", "2104.08663", "2305.06983", "2302.00083", "2303.07678"],
    "llms_scaling": ["2211.05100", "2112.11446", "2403.08295", "2204.06745", "2205.01068",
                     "2112.06905", "2309.16609"],
    "transformers_architecture": ["2002.05202", "2009.14794", "2007.14062", "2108.12409",
                                  "2001.04451", "2006.04768"],
    "generative_models": ["1710.10196", "1912.04958", "2102.09672", "2105.05233", "2205.11487",
                          "2204.06125", "1701.07875", "2206.00364"],
    "computer_vision": ["2103.14030", "2111.06377", "2201.03545", "2005.12872", "1608.06993",
                        "1704.04861", "1911.05722", "2002.05709"],
    "classic_deep_learning": ["1603.05027", "1611.05431", "1711.05101", "1709.01507", "1410.5401",
                              "1506.02025", "1505.00387"],
    "training_alignment": ["2204.05862", "2009.01325", "1909.08593", "2210.11416", "2401.10020"],
    "efficiency": ["2208.07339", "2309.06180", "1803.03635", "2211.17192", "2101.00190",
                   "2104.08691", "2211.10438", "1710.03740"],
    "embeddings": ["1310.4546", "2308.03281", "2212.09741", "1607.04606"],
    "prompting_reasoning": ["2303.17651", "2210.03493", "2210.03350", "2309.11495", "2211.10435",
                            "2310.06117"],
    "reinforcement_learning": ["1802.09477", "1511.05952", "1502.05477", "1509.02971", "1602.01783",
                               "1710.02298"],
    "agents_tools": ["2303.17580", "2305.15334", "2308.00352", "2308.08155"],
    "multimodal": ["2102.05918", "2205.01917", "2305.06500", "2308.12966", "2310.03744"],
    "evaluation": ["2103.03874", "2110.14168", "2210.09261", "1905.07830", "2107.03374"],
    "hallucination_factuality": ["2304.13734", "2203.11147", "2104.07567"],
    "interpretability_safety": ["2112.00861", "2207.05221", "2212.03827", "2202.03286"],
    "long_context": ["2309.17453", "2309.00071", "2307.02486"],
    "speech_audio": ["2106.07447", "2301.02111", "2005.08100", "1609.03499"],
    "classical_ml": ["1706.09516", "1802.03426", "1705.07874", "1602.04938"],
    "graph_neural_networks": ["1810.00826", "1704.01212", "1703.06103"],
}

# Round-3 additions: ~40 more canonical hand-picks (verified) + 6 gems mined from the
# old corpus that weren't already covered. Folded in the same way as CURATED.
CURATED_EXTRA = {
    "retrieval_rag": ["2304.09542", "2010.06467", "2407.13193", "2211.14876", "2205.04733"],
    "transformers_architecture": ["1904.10509"],
    "generative_models": ["1711.00937", "2212.09748", "2302.05543"],
    "computer_vision": ["2012.12877"],
    "training_alignment": ["2309.00267", "2405.14734", "2204.14146"],
    "efficiency": ["2305.13245", "2312.03863"],
    "embeddings": ["1803.11175", "2201.10005", "2003.07278"],
    "prompting_reasoning": ["2308.09687", "2210.11610"],
    "reinforcement_learning": ["1803.10122", "1912.01603"],
    "agents_tools": ["2307.16789", "2303.17760", "2308.03688"],
    "multimodal": ["2303.15343", "2305.05665", "2302.14045"],
    "evaluation": ["1905.00537", "2311.12022", "2307.03109"],
    "hallucination_factuality": ["2311.05232", "2309.01219"],
    "interpretability_safety": ["2210.07229", "2211.00593", "2309.08600"],
    "long_context": ["2309.12307", "2203.08913", "2311.12351"],
    "speech_audio": ["1512.02595", "1712.05884", "1904.08779", "2209.03143", "2205.10643"],
    "classical_ml": ["1603.06560", "1206.2944", "1702.08835", "1802.01933"],
    "graph_neural_networks": ["1606.09375", "1607.00653", "1611.07308", "2005.00687", "1901.00596"],
}


def strip_version(pid: str) -> str:
    return re.sub(r"v\d+$", "", pid.strip())


def cached_get(url: str) -> str:
    """GET a URL, caching the response body to disk. Sleeps only on cache miss."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(url.encode()).hexdigest()
    path = os.path.join(CACHE_DIR, f"{key}.xml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    time.sleep(ARXIV_RATE_LIMIT)
    return resp.text


def parse_entries(feed) -> list[dict]:
    """Convert Atom feed entries into project paper dicts (same shape build_index expects)."""
    out = []
    for entry in feed.entries:
        raw = entry.id
        pid = raw.split("/abs/")[1] if "/abs/" in raw else raw.strip()
        pid = strip_version(pid)
        title = entry.get("title", "").replace("\n", " ").strip()
        abstract = entry.get("summary", "").replace("\n", " ").strip()
        authors = [a.name for a in entry.get("authors", [])]
        published = entry.get("published", "")
        pdf_url = None
        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.href
                break
        if not pdf_url:
            pdf_url = f"http://arxiv.org/pdf/{pid}"
        categories = [t.term for t in entry.get("tags", [])]
        if not title:
            continue
        out.append({
            "paper_id": pid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published": published,
            "pdf_url": pdf_url,
            "categories": categories,
        })
    return out


def fetch_by_ids(ids: list[str]) -> dict[str, dict]:
    """Batch-fetch metadata for a list of arXiv IDs (chunks of 50)."""
    result = {}
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        url = f"{ARXIV_API}?id_list={','.join(batch)}&max_results=50"
        feed = feedparser.parse(cached_get(url))
        for p in parse_entries(feed):
            result[p["paper_id"]] = p
    return result


def search(keywords: str, max_results: int) -> list[dict]:
    """Relevance-sorted keyword search."""
    q = "all:" + "+".join(keywords.split())
    url = f"{ARXIV_API}?search_query={q}&sortBy=relevance&sortOrder=descending&max_results={max_results}"
    feed = feedparser.parse(cached_get(url))
    return parse_entries(feed)


def main():
    selected = {}          # paper_id -> manifest entry
    field_origin = {}      # field -> {"picked": n, "fill": n}

    # --- Hand-picks first (anchors + curated), guaranteed slots ---
    picks = {}
    for field in set(list(ANCHORS) + list(CURATED) + list(CURATED_EXTRA)):
        # dedupe within field, preserve order
        picks[field] = list(dict.fromkeys(
            ANCHORS.get(field, []) + CURATED.get(field, []) + CURATED_EXTRA.get(field, [])))
    all_pick_ids = [pid for ids in picks.values() for pid in ids]
    print(f"Fetching metadata for {len(all_pick_ids)} hand-picked papers...")
    pick_meta = fetch_by_ids(all_pick_ids)

    for field, ids in picks.items():
        field_origin.setdefault(field, {"picked": 0, "fill": 0})
        for pid in ids:
            if pid in selected:
                continue
            meta = pick_meta.get(pid)
            if not meta:
                print(f"  WARNING: hand-pick {pid} ({field}) not returned by arXiv")
                continue
            entry = dict(meta)
            entry["field"] = field
            entry["tier"] = "anchor" if pid in ANCHORS.get(field, []) else "curated"
            selected[pid] = entry
            field_origin[field]["picked"] += 1

    # --- Keyword fill per topic, independently, with quality gates ---
    for field, quota in FIELD_QUOTAS.items():
        field_origin.setdefault(field, {"picked": 0, "fill": 0})
        need = quota - field_origin[field]["picked"]
        if need <= 0:
            continue
        candidates = search(FIELD_KEYWORDS[field], max_results=300)
        added = 0
        for cand in candidates:
            if added >= need:
                break
            pid = cand["paper_id"]
            if pid in selected or pid in FILL_EXCLUDE:
                continue
            if not passes_filters(cand):
                continue
            entry = dict(cand)
            entry["field"] = field
            entry["tier"] = "fill"
            selected[pid] = entry
            field_origin[field]["fill"] += 1
            added += 1

    # --- Write manifest ---
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    manifest = list(selected.values())
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Report ---
    print("\n=== Corpus spread ===")
    print(f"{'topic':<28}{'quota':>6}{'picked':>8}{'fill':>6}{'total':>7}{'short':>7}")
    total_q = total_p = total_f = total_t = 0
    for field, quota in FIELD_QUOTAS.items():
        p = field_origin[field]["picked"]
        fl = field_origin[field]["fill"]
        t = p + fl
        short = quota - t
        total_q += quota; total_p += p; total_f += fl; total_t += t
        flag = "  <-- SHORT" if short > 0 else ""
        print(f"{field:<28}{quota:>6}{p:>8}{fl:>6}{t:>7}{short:>7}{flag}")
    print(f"{'TOTAL':<28}{total_q:>6}{total_p:>8}{total_f:>6}{total_t:>7}{total_q-total_t:>7}")
    print(f"\nManifest written: {MANIFEST_PATH}  ({len(manifest)} unique papers)")


if __name__ == "__main__":
    main()
