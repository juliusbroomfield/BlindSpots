import base64
import hashlib
import json
import os
import re
import sys
import time
from glob import glob
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

# verify requests is available
try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

# CONFIG

ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
DEFAULT_MODEL = "claude-sonnet-4-5"

# offline / cache behavior
OUTPUT_DIR = Path("reference_docs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USE_CACHE_ONLY = True                 # never re-download PDFs (assumes they exist in reference_docs/)
FETCH_HTML = False                    # skip HTML fetching by default

# PDF policy
MAX_BASE64_PDF_BYTES = 6 * 1024 * 1024   # <= 6 MB -> send as base64 document
MAX_BASE64_PAGES = 40                    # <= 40 pages -> allow base64
SEND_LARGE_PDFS_AS_TEXT = True           # >100 pages -> send local text excerpt instead of PDF
PDF_TEXT_SNIPPET_CHARS = 12000           # cap text excerpt length
PDF_CHUNK_SIZE_CHARS = 80000             # max chars per chunk when splitting large PDFs
PDFS_PER_BATCH = 1                       # 1 per call (keeps TPM low)

# retries. the anthropic sdk backs off using the server's Retry-After
# header, so there's no hand-rolled throttle here
RETRY_MAX = 5

# model output size
MAX_OUTPUT_TOKENS = 500

# HTML snippet budgets
MAX_HTML_TOTAL_CHARS = 3000
MAX_HTML_PER_DOC_CHARS = 1000

# Output JSON file (single file)
OUTPUT_JSON = Path("hidden_needs.json")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HiddenNeeds/offline-1.3"

# SOURCES

COUNTRY_BUSINESS_SLUGS = {
    "Australia": "australian",
    "Brazil": "brazilian",
    "Canada": "canadian",
    "China": "chinese",
    "India": "indian",
    "Japan": "japanese",
    "Mexico": "mexican",
    "South Africa": "south-african",
    "Germany": "german",
    "United States": "american",
}

def build_cultural_atlas_urls(slugs: dict) -> list:
    base = "https://culturalatlas.sbs.com.au"
    return [f"{base}/{slug}-culture/{slug}-culture-business-culture" for slug in slugs.values()]

SOURCES: dict[tuple[str, str], dict[str, list[str]]] = {
    ("religion", "healthcare"): {
        "pdfs": ["https://www.yorkhospitals.nhs.uk/seecmsfile/?id=597"],
        "html": [],
    },
    ("religion", "food"): {
        "pdfs": [
            "https://food.unt.edu/documents/J.pdf",
            "https://food.unt.edu/documents/I.pdf",
            "https://food.unt.edu/documents/H.pdf",
            "https://food.unt.edu/documents/C.pdf",
            "https://food.unt.edu/documents/B.pdf",
        ],
        "html": [],
    },
    ("bodily condition", "architectural design"): {
        "pdfs": ["https://www.nyc.gov/html/ddc/downloads/pdf/udny/udny2.pdf"],
        "html": [],
    },
    ("bodily condition", "technological design"): {
        "pdfs": [],
        "html": ["https://www.w3.org/TR/WCAG22/#abstract"],
    },
    ("culture", "business"): {
        "pdfs": [],
        "html": build_cultural_atlas_urls(COUNTRY_BUSINESS_SLUGS),
    },
    ("language", "healthcare"): {
        "pdfs": ["https://healthlaw.org/wp-content/uploads/2018/09/Federal-Language-Access-Laws.pdf"],
        "html": [
            "https://www.hhs.gov/ohrp/regulations-and-policy/guidance/obtaining-and-documenting-infomed-consent-non-english-speakers/index.html",
            "https://www.hhs.gov/civil-rights/for-individuals/faqs/may-an-lep-person-use-a-family-member-as-an-interpreter/709/index.html",
            "https://www.hhs.gov/civil-rights/for-individuals/section-1557/1557faqs/aggregation_tagline/index.html",
            "https://cmelearning.com/new-2016-aca-rules-significantly-affect-the-law-of-language-access/",
        ],
    },
    ("socioeconomic status", "transportation"): {
        "pdfs": [
            "https://transitcenter.org/wp-content/uploads/2021/09/Equity-in-Practice_web.pdf",
            "https://www.apta.com/wp-content/uploads/APTA_Late-Shift_Report.pdf",
        ],
        "html": [],
    },
    ("socioeconomic status", "public facility locations"): {
        "pdfs": ["https://www.dol.gov/sites/dolgov/files/ETA/advisories/TEGL/2017/TEGL_16-16.pdf"],
        "html": [],
    },
    ("socioeconomic status", "work"): {
        "pdfs": ["https://www.apta.com/wp-content/uploads/APTA_Late-Shift_Report.pdf"],
        "html": [],
    },
    ("socioeconomic status", "education"): {
        "pdfs": ["https://www.nea.org/sites/default/files/2020-10/NEA%20Report%20-%20Digital%20Equity%20for%20Students%20and%20Educators_0.pdf"],
        "html": [],
    },
}

# DEPS + HELPERS

def ensure_dependencies():
    try:
        import anthropic  # noqa: F401
        import pdfplumber  # noqa: F401
        if FETCH_HTML:
            import bs4  # noqa: F401
        import requests as _rq  # noqa: F401
    except ImportError:
        import subprocess
        pkgs = ["anthropic", "pdfplumber", "requests"]
        if FETCH_HTML:
            pkgs.append("beautifulsoup4")
        print("Installing dependencies:", ", ".join(pkgs))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

def _safe_filename_from_url(url: str, fallback_stem: str = "document") -> str:
    p = urlparse(url)
    path_name = unquote(Path(p.path).name) if p.path else ""
    stem = (Path(path_name).stem or "").strip()
    if not stem:
        q = parse_qs(p.query)
        if "id" in q and q["id"]:
            base = Path(p.path).stem or "seecmsfile"
            stem = f"{base}_id_{q['id'][0]}"
        else:
            host = (p.netloc or "host").replace(".", "_")
            path_slug = "_".join(s for s in p.path.split("/") if s) or "file"
            stem = f"{host}_{path_slug}"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("_")
    if not safe:
        safe = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return f"{safe}.pdf"

def _safe_stem(url: str) -> str:
    fn = _safe_filename_from_url(url)
    return Path(fn).stem

def locate_cached_pdf(url: str) -> Path | None:
    """find the cached PDF for a URL, including variant filenames (e.g., J_7.pdf)."""
    expected = OUTPUT_DIR / _safe_filename_from_url(url)
    if expected.exists() and expected.stat().st_size > 0:
        return expected
    # variant search by stem prefix (handles earlier runs that added suffixes like _7)
    stem = _safe_stem(url)
    pattern = str(OUTPUT_DIR / f"{stem}*.pdf")
    candidates = [Path(p) for p in glob(pattern)]
    # if none, try loose match on key tokens (helps york hospitals special-case)
    if not candidates:
        p = urlparse(url)
        tokens = [t for t in [Path(p.path).stem, p.netloc.split(".")[0]] if t]
        for t in tokens:
            candidates.extend(Path(OUTPUT_DIR).glob(f"*{t}*.pdf"))
        # unique
        candidates = list({c.resolve() for c in candidates})
    if not candidates:
        return None
    # pick the largest (likely the real doc)
    candidates.sort(key=lambda fp: (fp.stat().st_size, fp.stat().st_mtime), reverse=True)
    return candidates[0]

def download_pdf(url: str, dest_dir: Path, retries: int = 2, backoff: float = 1.5) -> Path | None:
    """download only if file missing (respects USE_CACHE_ONLY)."""
    cached = locate_cached_pdf(url)
    if cached:
        return cached
    if USE_CACHE_ONLY:
        print(f"Cache-only mode; not downloading {url}")
        return None

    dest_path = OUTPUT_DIR / _safe_filename_from_url(url)
    attempt = 0
    while attempt < retries:
        try:
            with requests.get(
                url,
                headers={"User-Agent": UA, "Accept": "application/pdf,*/*;q=0.8"},
                stream=True,
                timeout=60,
                allow_redirects=True,
            ) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            return dest_path
        except Exception as e:
            attempt += 1
            print(f"Failed to download {url} (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(backoff ** attempt)
    return None

def pdf_stats(pdf_path: Path) -> tuple[int, int]:
    """return (pages, size_bytes)."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = len(pdf.pages)
    except Exception:
        pages = 0
    size = pdf_path.stat().st_size if pdf_path.exists() else 0
    return pages, size

def pdf_to_snippet(pdf_path: Path, char_limit: int = PDF_TEXT_SNIPPET_CHARS) -> str:
    """extract a capped amount of text locally (no network)."""
    try:
        import pdfplumber
        parts: list[str] = []
        total = 0
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if total >= char_limit:
                    break
                txt = (page.extract_text() or "").strip()
                if not txt:
                    continue
                remaining = char_limit - total
                if len(txt) > remaining:
                    txt = txt[:remaining]
                parts.append(txt)
                total += len(txt)
        return "\n\n".join(parts)
    except Exception as e:
        print(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def fetch_html_text(url: str) -> str:
    if not FETCH_HTML:
        return ""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(r.content, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception as e:
        print(f"Failed to fetch or parse {url}: {e}")
        return ""

def collect_html_snippets(html_urls: list[str]) -> list[str]:
    total = 0
    out: list[str] = []
    for u in html_urls:
        if total >= MAX_HTML_TOTAL_CHARS:
            break
        txt = fetch_html_text(u)
        if not txt:
            continue
        cut = txt[:MAX_HTML_PER_DOC_CHARS]
        out.append(cut)
        total += len(cut)
    return out

# JSON EXTRACTION (tolerant)

def _try_json_load(s: str):
    try:
        return json.loads(s), None
    except Exception as e:
        return None, e

def _strip_code_fence(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.DOTALL)

def _extract_json_array(s: str) -> list:
    s0 = s.strip()
    obj, _ = _try_json_load(s0)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("items", "data", "results", "hidden_needs", "needs"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
    s1 = _strip_code_fence(s0)
    obj, _ = _try_json_load(s1)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("items", "data", "results", "hidden_needs", "needs"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
    # balanced scan
    text = s1
    n = len(text)
    i = 0
    while i < n:
        if text[i] == '[':
            depth, j = 1, i + 1
            while j < n and depth > 0:
                c = text[j]
                if c == '[':
                    depth += 1
                elif c == ']':
                    depth -= 1
                j += 1
            if depth == 0:
                candidate = text[i:j]
                obj, _ = _try_json_load(candidate)
                if isinstance(obj, list):
                    return obj
                i = j
                continue
            break
        i += 1
    return []

# TOKEN RATE LIMITER




def estimate_input_tokens_for_message(message_content: list[dict], pdf_pages_hint: int = 0) -> int:
    """
    very rough estimate:
      - text blocks: ~1 token per 4 chars
      - PDFs: ~600 tokens per page (conservative)
    """
    text_chars = sum(len(b.get("text", "")) for b in message_content if b.get("type") == "text")
    text_tokens = text_chars // 4
    doc_blocks = sum(1 for b in message_content if b.get("type") == "document")
    doc_tokens = (pdf_pages_hint * 600) if doc_blocks else 0
    return max(1, text_tokens + doc_tokens)

# CLAUDE CALL

def call_claude(messages, system, model=DEFAULT_MODEL, temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS, est_input_tokens: int = 0):
    """
    one anthropic call.

    no retry loop and no throttle: the sdk already backs off using the
    retry-After header the server sends on a 429, with jitter, which is better
    information than any sleep we could guess at.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("set ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key, max_retries=RETRY_MAX)
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    )


def anthropic_text(resp) -> str:
    if resp is None:
        return ""
    parts = []
    try:
        for block in resp.content:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
    except Exception:
        content = getattr(resp, "content", None) or resp.get("content", [])
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
    return "\n".join(parts).strip()

# PROMPTS

SYSTEM_PROMPT = (
    "Return ONLY a JSON array of hidden needs grounded in the provided sources. "
    "If none are supported, return []. "
    "Each item: {name, applies_to, trigger_cues, scope, severity, citations[{org,title,url,quoted_support}]}. "
    "Prefer citing the PDF URL when available. No prose, no code fences."
)

def build_msg_content(perspective: str, domain: str, pdf_blocks: list[dict], html_snippets: list[str]) -> list[dict]:
    content: list[dict] = [
        {"type": "text", "text": f"Perspective: {perspective}\nDomain: {domain}\nUse ONLY the following sources:"}
    ]
    content.extend(pdf_blocks)
    if html_snippets:
        content.append({"type": "text", "text": "Additional short text snippets:"})
        for snip in html_snippets:
            content.append({"type": "text", "text": snip[:MAX_HTML_PER_DOC_CHARS]})
    content.append({"type": "text", "text": "Now output ONLY the JSON array as specified."})
    return content

def make_pdf_block_from_url(url: str) -> dict:
    return {"type": "document", "source": {"type": "url", "url": url}}

def make_pdf_block_from_base64(b64: str) -> dict:
    return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}}

def make_text_block_from_pdf(pdf_path: Path) -> dict:
    text = pdf_to_snippet(pdf_path, PDF_TEXT_SNIPPET_CHARS)
    return {"type": "text", "text": f"[EXCERPT from {pdf_path.name}]\n\n{text}"}

def pdf_to_chunks(pdf_path: Path, chunk_size: int = PDF_CHUNK_SIZE_CHARS) -> list[str]:
    """extract PDF text and split into multiple chunks up to chunk_size each."""
    try:
        import pdfplumber
        chunks: list[str] = []
        current_parts: list[str] = []
        current_size = 0
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = (page.extract_text() or "").strip()
                if not txt:
                    continue
                if current_size + len(txt) > chunk_size and current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_size = 0
                current_parts.append(txt)
                current_size += len(txt)
        if current_parts:
            chunks.append("\n\n".join(current_parts))
        return chunks
    except Exception as e:
        print(f"Failed to extract text chunks from {pdf_path}: {e}")
        return []

def make_text_blocks_from_pdf_chunks(pdf_path: Path) -> list[dict]:
    """turn chunked text from a PDF into a list of text blocks."""
    chunks = pdf_to_chunks(pdf_path, PDF_CHUNK_SIZE_CHARS)
    blocks: list[dict] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        label = f" (part {idx}/{total})" if total > 1 else ""
        blocks.append({"type": "text", "text": f"[EXCERPT from {pdf_path.name}{label}]\n\n{chunk}"})
    return blocks

def plan_pdf_blocks_from_cache(pdf_urls: list[str]) -> tuple[list[tuple[dict, int]], int]:
    """
    for each URL, use cached PDF if present. choose:
      - base64 (<= 40 pages and <= 6MB)
      - text excerpts/chunks if pages > 50 and SEND_LARGE_PDFS_AS_TEXT
      - url block otherwise (no local re-download)
    returns ([(block, pages)], total_pages_hint)
    """
    total_pages_hint = 0
    blocks_and_pages: list[tuple[dict, int]] = []
    for url in pdf_urls:
        pdf_path = locate_cached_pdf(url)
        if not pdf_path:
            if USE_CACHE_ONLY:
                print(f"Cache-only mode: missing cached file for {url}; skipping this source.")
                continue
        if not pdf_path:
            pdf_path = download_pdf(url, OUTPUT_DIR)
            if not pdf_path:
                print(f"Could not obtain PDF for {url}; skipping.")
                continue

        pages, size = pdf_stats(pdf_path)
        total_pages_hint += max(0, pages)

        if pages == 0:
            block = make_pdf_block_from_url(url)
            blocks_and_pages.append((block, 0))
            continue

        if SEND_LARGE_PDFS_AS_TEXT and pages > 50:
            text_blocks = make_text_blocks_from_pdf_chunks(pdf_path)
            for tb in text_blocks:
                blocks_and_pages.append((tb, 0))  # text blocks -> no page hint
            continue

        if pages <= MAX_BASE64_PAGES and 0 < size <= MAX_BASE64_PDF_BYTES:
            try:
                b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode("utf-8")
                block = make_pdf_block_from_base64(b64)
                blocks_and_pages.append((block, pages))
            except Exception as e:
                print(f"Base64 read failed for {pdf_path}: {e}; using text chunks.")
                text_blocks = make_text_blocks_from_pdf_chunks(pdf_path)
                for tb in text_blocks:
                    blocks_and_pages.append((tb, 0))
        else:
            block = make_pdf_block_from_url(url)
            blocks_and_pages.append((block, pages))

    return blocks_and_pages, total_pages_hint

# AGGREGATION (YOUR SCHEMA)

def _pick_pdf_url_from_citations(citations: list[dict]) -> str | None:
    """prefer a citation URL that looks like a PDF; else first URL; else None."""
    if not isinstance(citations, list):
        return None
    pdf_like = [c.get("url") for c in citations if isinstance(c, dict) and isinstance(c.get("url"), str) and c.get("url", "").lower().endswith(".pdf")]
    if pdf_like:
        return pdf_like[0]
    any_url = [c.get("url") for c in citations if isinstance(c, dict) and isinstance(c.get("url"), str)]
    return any_url[0] if any_url else None

def _first_quote(citations: list[dict]) -> str | None:
    if not isinstance(citations, list):
        return None
    for c in citations:
        if isinstance(c, dict) and isinstance(c.get("quoted_support"), str) and c["quoted_support"].strip():
            return c["quoted_support"].strip()
    return None

def simplify_item_to_schema(item: dict, default_source_hint: str | None = None) -> dict | None:
    """Map model item -> {name, source, quoted_support}."""
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    citations = item.get("citations", [])
    src = _pick_pdf_url_from_citations(citations)
    quote = _first_quote(citations)
    if not src:
        src = default_source_hint
    return {
        "name": name.strip(),
        "source": src if src else "",
        "quoted_support": quote if isinstance(quote, str) else ""
    }

def add_to_aggregate(agg: dict, domain: str, perspective: str, simplified_items: list[dict]):
    agg.setdefault(domain, {}).setdefault(perspective, [])
    # dedupe by (name, source, quote)
    seen = {(n["name"], n.get("source", ""), n.get("quoted_support", "")) for n in agg[domain][perspective]}
    for n in simplified_items:
        key = (n["name"], n.get("source", ""), n.get("quoted_support", ""))
        if key not in seen:
            agg[domain][perspective].append(n)
            seen.add(key)

# PIPELINE

def generate_hidden_needs_for_pair(perspective: str, domain: str, pdf_urls: list[str], html_urls: list[str]) -> list[dict]:
    html_snips = collect_html_snippets(html_urls) if FETCH_HTML else []
    blocks_and_pages, _ = plan_pdf_blocks_from_cache(pdf_urls)

    if not blocks_and_pages and not html_snips:
        print(f"No sources for {perspective}/{domain}; skipping.")
        return []

    all_items: list[dict] = []
    for i in range(0, len(blocks_and_pages) or 1, PDFS_PER_BATCH):
        batch_pairs = blocks_and_pages[i:i+PDFS_PER_BATCH] if blocks_and_pages else []
        batch_blocks = [bp[0] for bp in batch_pairs]
        batch_pages_hint = sum(bp[1] for bp in batch_pairs)  # <-- use per-batch pages only

        # Default source hint (filename) to fill in if the model omits a URL
        default_source_hint = None
        if batch_blocks:
            src = batch_blocks[0].get("source", {})
            if isinstance(src, dict) and "url" in src:
                cached = locate_cached_pdf(src["url"])
                default_source_hint = cached.name if cached else Path(_safe_filename_from_url(src["url"])).name
            elif isinstance(src, dict) and src.get("type") == "base64":
                default_source_hint = "embedded_base64.pdf"

        msg_content = build_msg_content(perspective, domain, batch_blocks, html_snips)
        est_input_tokens = estimate_input_tokens_for_message(msg_content, pdf_pages_hint=batch_pages_hint)

        try:
            resp = call_claude(
                messages=[{"role": "user", "content": msg_content}],
                system=SYSTEM_PROMPT,
                model=DEFAULT_MODEL,
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
                est_input_tokens=est_input_tokens
            )
            text = anthropic_text(resp)
            arr = _extract_json_array(text)
        except Exception as e:
            msg = str(e)
            if ("Could not process PDF" in msg or "maximum of 100 PDF pages" in msg) and batch_blocks:
                print("Retrying batch with text-only excerpts due to PDF processing/page-limit error...")
                # build a new message that uses local text excerpts instead of documents
                text_blocks = []
                for b in batch_blocks:
                    src = b.get("source", {})
                    if isinstance(src, dict) and "url" in src:
                        cached = locate_cached_pdf(src["url"])
                        if cached:
                            text_blocks.append(make_text_block_from_pdf(cached))
                        else:
                            text_blocks.append({"type": "text", "text": "[PDF excerpt omitted]"})
                    else:
                        text_blocks.append({"type": "text", "text": "[PDF excerpt omitted]"})
                msg_content2: list[dict] = [
                    {"type": "text", "text": f"Perspective: {perspective}\nDomain: {domain}\nUse ONLY the following excerpts and snippets:"}
                ] + text_blocks
                if html_snips:
                    msg_content2.append({"type": "text", "text": "Additional short text snippets:"})
                    for snip in html_snips:
                        msg_content2.append({"type": "text", "text": snip[:MAX_HTML_PER_DOC_CHARS]})
                msg_content2.append({"type": "text", "text": "Now output ONLY the JSON array as specified."})

                try:
                    resp2 = call_claude(
                        messages=[{"role": "user", "content": msg_content2}],
                        system=SYSTEM_PROMPT,
                        model=DEFAULT_MODEL,
                        temperature=0.0,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        est_input_tokens=estimate_input_tokens_for_message(msg_content2)
                    )
                    text2 = anthropic_text(resp2)
                    arr = _extract_json_array(text2)
                except Exception as e2:
                    print(f"Batch error after text-fallback: {e2}")
                    arr = []
            else:
                print(f"Batch error: {e}")
                arr = []

        # simplify to your schema
        simplified: list[dict] = []
        for item in arr:
            if isinstance(item, dict):
                simp = simplify_item_to_schema(item, default_source_hint=default_source_hint)
                if simp:
                    simplified.append(simp)

        all_items.extend(simplified)

    return all_items

# MAIN

def main():
    ensure_dependencies()
    # aggregate to your schema: { "<domain>": { "<perspective>": [ {name, source, quoted_support} ] } }
    aggregate: dict[str, dict[str, list[dict]]] = {}

    for (perspective, domain), entry in SOURCES.items():
        print(f"\nProcessing {perspective} / {domain}...")
        pdf_urls = entry.get("pdfs", [])
        html_urls = entry.get("html", [])
        try:
            simplified_items = generate_hidden_needs_for_pair(perspective, domain, pdf_urls, html_urls)
            add_to_aggregate(aggregate, domain, perspective, simplified_items)
            print(f"Added {len(simplified_items)} items into {domain} -> {perspective}.")
        except Exception as e:
            print(f"Error generating hidden needs for {perspective}/{domain}: {e}")

    OUTPUT_JSON.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUTPUT_JSON.resolve()}")

if __name__ == "__main__":
    main()