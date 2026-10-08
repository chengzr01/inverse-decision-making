"""Download Supreme Court cases from Justia into raw files and structured JSON.

Run in the gym environment:
    python datasets/law/download.py [CASE_URL ...]

Optional PDF extraction requires pypdf (see requirements.txt). Use --html-file
for browser-saved HTML, or --pdf-url when the case page is inaccessible.
Output defaults to directories alongside this script, regardless of cwd.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

url = ["https://supreme.justia.com/cases/federal/us/576/644/"]
ROOT = Path(__file__).resolve().parent
CASE_PATH = re.compile(r"^/cases/federal/us/([\w-]+)/([\w-]+)/?$")
SECTIONS = {"Justia Summary", "Annotation", "Primary Holding", "Facts", "Attorneys", "Opinions",
            "Case Commentary", "Syllabus", "Materials", "Procedural History",
            "Search This Case", "U.S. Supreme Court Resources"}
DECLARATION = (
    r"(?:Mr\.?\s+)?(?:Chief\s+)?Justice\s+"
    r"(?P<author>[A-Za-z’'-]+)"
    r"(?P<kind>\s+(?:delivered\s+the\s+opinion|announced\s+the\s+judgment)[^.]*|"
    r",\s*(?:with\s+whom\b.{0,250}?\bjoin(?:s|ed)?\s*,\s*)?"
    r"(?:dissenting|concurring)[^.]*?)\."
)
AUTHOR = re.compile(r"^" + DECLARATION, re.IGNORECASE | re.MULTILINE)


def case_id(source_url: str) -> str:
    parsed = urlsplit(source_url)
    match = CASE_PATH.fullmatch(parsed.path)
    if parsed.scheme != "https" or parsed.hostname != "supreme.justia.com" or not match:
        raise ValueError(f"Expected a Justia Supreme Court case URL: {source_url}")
    return "us_" + "_".join(match.groups())


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "SupremeCourtDataset/1.0 (research downloader)"
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch(session: requests.Session, source_url: str, timeout: float) -> bytes:
    response = session.get(source_url, timeout=timeout)
    response.raise_for_status()
    return response.content


def save(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def lines_of(node) -> list[str]:
    """Preserve paragraphs while keeping inline citations and names together."""
    copy = BeautifulSoup(str(node), "html.parser")
    for tag in copy.find_all(["script", "style", "nav", "footer", "button"]):
        tag.decompose()
    for tag in copy.find_all(["p", "div", "section", "article", "li", "ul", "ol",
                              "h1", "h2", "h3", "h4", "h5", "h6", "br", "hr"]):
        tag.insert_before("\n")
        tag.insert_after("\n")
    return [re.sub(r"\s+", " ", line).strip()
            for line in copy.get_text().splitlines() if line.strip()]


def section(lines: list[str], name: str, start: int = 0) -> str | None:
    try:
        begin = lines.index(name, start) + 1
    except ValueError:
        return None
    end = next((i for i in range(begin, len(lines)) if lines[i] in SECTIONS), len(lines))
    return "\n\n".join(lines[begin:end]).strip() or None


def split_opinions(text: str | None) -> list[dict]:
    if not text:
        return []
    matches = list(AUTHOR.finditer(text))
    if not matches:
        return [{"type": "unknown", "author": None, "text": text}]
    result = []
    for i, match in enumerate(matches):
        kind = match["kind"].lower()
        opinion_type = ("dissent" if "dissenting" in kind else
                        "concurrence" if "concurring" in kind else
                        "plurality" if "announced" in kind else "majority")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result.append({"type": opinion_type, "author": match["author"].strip(),
                       "text": text[0 if i == 0 else match.start():end].strip()})
    return result


def new_record(source_url: str) -> dict:
    parts = CASE_PATH.fullmatch(urlsplit(source_url).path)
    identifier = case_id(source_url)
    return {
        "schema_version": 1, "case_id": identifier, "source_url": source_url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "court": "Supreme Court of the United States", "case_name": None,
        "citation": f"{parts[1]} U.S. {parts[2]}" if parts[2].isdigit() else None,
        "year": None, "docket_numbers": [],
        "dates": {"granted": None, "argued": None, "decided": None},
        "justia_summary": None,
        "annotation": {"primary_holding": None, "facts": None, "attorneys": [],
                       "opinion_summaries": [], "commentary": None},
        "syllabus": None, "opinions": [], "procedural_history": None,
        "materials": [], "pdf_urls": [], "files": {}, "warnings": [],
    }


def parse_html(html: bytes | str, source_url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1")
    if heading is None or not re.search(r"\bU\.?\s*S\.?\s+", heading.get_text(" ")):
        raise ValueError("Missing Supreme Court case heading (possibly an access-block page)")
    record = new_record(source_url)
    title = heading.get_text(" ", strip=True)
    match = re.match(r"(.+?),\s*(\d+\s+U\.?\s*S\.?\s+\S+)\s*\((\d{4})\)", title)
    if match:
        record.update(case_name=match[1], citation=match[2], year=int(match[3]))
    else:
        record["case_name"] = title
    lines = lines_of(soup.find("main") or soup)
    start = next((i for i, line in enumerate(lines) if line == title), 0)
    lines = lines[start:]
    end = next((i for i, line in enumerate(lines) if line in
                {"Search This Case", "U.S. Supreme Court Resources"}), len(lines))
    lines = lines[:end]
    header = "\n".join(lines[:next((i for i, line in enumerate(lines)
                                     if line in SECTIONS), len(lines))])
    for key in record["dates"]:
        found = re.search(key + r"\s*:\s*([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", header, re.I)
        if found:
            record["dates"][key] = datetime.strptime(found[1], "%B %d, %Y").date().isoformat()
    record["docket_numbers"] = list(dict.fromkeys(re.findall(
        r"\b\d{1,3}-\d{1,5}\b", header.replace("–", "-"))))
    record["justia_summary"] = section(lines, "Justia Summary")
    for key, label in [("primary_holding", "Primary Holding"), ("facts", "Facts"),
                       ("commentary", "Case Commentary")]:
        record["annotation"][key] = section(lines, label)
    attorneys = section(lines, "Attorneys")
    if attorneys:
        record["annotation"]["attorneys"] = attorneys.split("\n\n")
    syllabus_index = next((i for i, line in enumerate(lines) if line == "Syllabus"), len(lines))
    summaries = section(lines[:syllabus_index], "Opinions")
    if summaries and any(line in {"Primary Holding", "Facts", "Case Commentary"} for line in lines[:syllabus_index]):
        chunks = re.split(r"(?:^|\n\n)(Majority|Dissent|Concurrence|Plurality)\n\n", summaries)
        record["annotation"]["opinion_summaries"] = [
            {"type": chunks[i].lower(), "text": chunks[i + 1].strip()}
            for i in range(1, len(chunks) - 1, 2)]
    if syllabus_index < len(lines):
        # 'Syllabus' is also repeated inside the court's text.
        end = next((i for i in range(syllabus_index + 1, len(lines))
                    if lines[i] in SECTIONS - {"Syllabus"}), len(lines))
        record["syllabus"] = "\n\n".join(lines[syllabus_index + 1:end]) or None
        opinion_text = section(lines, "Opinions", end)
    else:
        opinion_text = section(lines, "Opinions")
    if not opinion_text:
        container = soup.select_one("#opinion, .opinions, .case-opinion")
        if container:
            opinion_text = "\n\n".join(lines_of(container))
    record["opinions"] = split_opinions(opinion_text)
    record["procedural_history"] = section(lines, "Procedural History")
    for link in soup.find_all("a", href=True):
        href = urljoin(source_url, link["href"])
        label = link.get_text(" ", strip=True)
        parsed = urlsplit(href)
        if parsed.scheme not in {"http", "https"}:
            continue
        # Exclude PDFs cited as evidence rather than court documents.
        if "download pdf" in label.lower() or (parsed.hostname == "supreme.justia.com" and parsed.path.endswith("/case.pdf")):
            if href not in record["pdf_urls"]:
                record["pdf_urls"].append(href)
        if parsed.hostname in {"oyez.org", "www.oyez.org"}:
            item = {"title": label, "url": href}
            if item not in record["materials"]:
                record["materials"].append(item)
    if not record["opinions"]:
        record["warnings"].append("No judicial opinion extracted from HTML; check source layout or PDFs.")
    return record


def parse_pdf(data: bytes) -> dict:
    from pypdf import PdfReader

    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError("Downloaded document is not a PDF")
    reader = PdfReader(BytesIO(data))
    texts = [page.extract_text() or "" for page in reader.pages]
    if not any(text.strip() for text in texts):
        raise ValueError("PDF has no extractable text; OCR is required")
    boundary = next((i for i in range(1, len(texts))
                     if re.search(r"Opinion of (?:the Court|.+?, J\.)", texts[i][:500])
                     and "Syllabus" not in texts[i][:500]), None)
    syllabus = "\n\n".join(texts[:boundary]) if boundary and "Syllabus" in texts[0] else None
    judicial = "\n\n".join(texts[boundary:] if syllabus else texts)
    normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", judicial)
    normalized = re.sub(r"\s+(?=" + DECLARATION + r")", "\n", normalized, flags=re.I)
    metadata = {"case_name": None, "docket_numbers": [], "dates": {}, "year": None}
    title = str((reader.metadata or {}).get("/Title", ""))
    title_match = re.match(r"(\d+-\d+)\s+(.+?)\s*\((\d{2}/\d{2}/\d{4})\)", title)
    if title_match:
        metadata["case_name"] = title_match[2]
        metadata["docket_numbers"] = [title_match[1]]
        metadata["dates"]["decided"] = datetime.strptime(title_match[3], "%m/%d/%Y").date().isoformat()
        metadata["year"] = int(title_match[3][-4:])
    header = re.sub(r"\s+", " ", texts[0])
    for key in ["argued", "decided"]:
        match = re.search(key + r"\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", header, re.I)
        if match:
            metadata["dates"][key] = datetime.strptime(match[1], "%B %d, %Y").date().isoformat()
    return {"page_count": len(texts), "metadata": metadata, "text": "\n\n".join(texts),
            "syllabus": syllabus, "opinions": split_opinions(normalized)}


def download_case(source_url: str, output_dir: Path = ROOT, *,
                  session: requests.Session | None = None, timeout: float = 60,
                  refresh: bool = False, html_file: Path | None = None,
                  pdf_url: str | None = None, download_pdfs: bool = True) -> dict:
    if session is None:
        with make_session() as owned:
            return download_case(source_url, output_dir, session=owned, timeout=timeout,
                                 refresh=refresh, html_file=html_file, pdf_url=pdf_url,
                                 download_pdfs=download_pdfs)
    identifier = case_id(source_url)
    raw_dir = output_dir / "raw" / identifier
    html_path = raw_dir / "case.html"
    try:
        html = (html_file.read_bytes() if html_file else
                html_path.read_bytes() if html_path.exists() and not refresh else
                fetch(session, source_url, timeout))
        record = parse_html(html, source_url)
        save(html_path, html)
        record["files"]["html"] = str(html_path.relative_to(output_dir))
    except (requests.RequestException, ValueError) as exc:
        if not pdf_url:
            raise RuntimeError(f"Cannot obtain case HTML: {exc}. Use --html-file with a "
                               "browser-saved page or --pdf-url with its opinion PDF.") from exc
        record = new_record(source_url)
        record["warnings"].append(f"HTML unavailable; editorial sections and metadata may be missing: {exc}")
    if pdf_url and pdf_url not in record["pdf_urls"]:
        record["pdf_urls"].append(pdf_url)
    record["pdf_documents"] = []
    if download_pdfs:
        for i, pdf_source in enumerate(record["pdf_urls"], 1):
            pdf_path = raw_dir / f"opinion_{i}.pdf"
            try:
                data = (pdf_path.read_bytes() if pdf_path.exists() and not refresh else
                        fetch(session, pdf_source, timeout))
                if not data.lstrip().startswith(b"%PDF-"):
                    raise ValueError("Downloaded document is not a PDF")
                save(pdf_path, data)
                document = {"url": pdf_source, "file": str(pdf_path.relative_to(output_dir))}
                record["pdf_documents"].append(document)
                extracted = parse_pdf(data)
                text_path = pdf_path.with_suffix(".txt")
                save(text_path, extracted.pop("text").encode("utf-8"))
                document.update(text_file=str(text_path.relative_to(output_dir)), **extracted)
                for key in ["case_name", "year", "docket_numbers"]:
                    if not record[key] and extracted["metadata"][key]:
                        record[key] = extracted["metadata"][key]
                for key, value in extracted["metadata"]["dates"].items():
                    if not record["dates"][key]:
                        record["dates"][key] = value
                if not record["opinions"]:
                    record["opinions"] = extracted["opinions"]
                if not record["syllabus"]:
                    record["syllabus"] = extracted["syllabus"]
            except Exception as exc:
                record["warnings"].append(f"PDF {pdf_source}: {exc}")
    if not record["opinions"] and not record["syllabus"] and not record["justia_summary"]:
        raise RuntimeError("No substantive content extracted; no JSON record written")
    destination = output_dir / "processed" / f"{identifier}.json"
    save(destination, (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="*", help="Justia case URLs (defaults to url list)")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--html-file", type=Path, help="Browser-saved HTML (one case only)")
    parser.add_argument("--pdf-url", help="Opinion PDF URL, also used if HTML is blocked (one case only)")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Download again instead of using cached files")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--delay", type=float, default=1, help="Seconds between cases")
    args = parser.parse_args(argv)
    sources = list(dict.fromkeys(args.urls or url))
    if (args.html_file or args.pdf_url) and len(sources) != 1:
        parser.error("--html-file and --pdf-url require exactly one case URL")
    if args.timeout <= 0 or args.delay < 0:
        parser.error("timeout must be positive and delay nonnegative")
    if args.pdf_url and args.no_pdf:
        parser.error("--pdf-url cannot be combined with --no-pdf")
    failures = 0
    with make_session() as session:
        for i, source in enumerate(sources):
            if i:
                time.sleep(args.delay)
            try:
                record = download_case(source, args.output_dir, session=session,
                                       timeout=args.timeout, refresh=args.refresh,
                                       html_file=args.html_file, pdf_url=args.pdf_url,
                                       download_pdfs=not args.no_pdf)
                print(f"Saved {args.output_dir / 'processed' / (record['case_id'] + '.json')}")
                for warning in record["warnings"]:
                    print(f"Warning: {warning}", file=sys.stderr)
            except Exception as exc:
                failures += 1
                print(f"Failed {source}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
