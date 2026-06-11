import time
import os
import feedparser
import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT = 3


def fetch_papers(category_query: str, max_papers: int) -> list[dict]:
    """
    Fetch paper metadata from arXiv API.

    Paginates through results in batches of 50.
    Enforces 3-second delay between requests.

    Args:
        category_query: e.g. "cat:cs.CL OR cat:cs.LG"
        max_papers: total papers to fetch (e.g. 200)

    Returns list of dicts, each with keys:
        - paper_id: str
        - title: str
        - abstract: str
        - authors: list[str]
        - published: str (ISO date)
        - pdf_url: str
        - categories: list[str]
    """
    papers = []
    batch_size = 50
    fetched = 0

    while fetched < max_papers:
        remaining = max_papers - fetched
        batch = min(batch_size, remaining)

        params = {
            "search_query": category_query,
            "start": fetched,
            "max_results": batch,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        response.raise_for_status()

        feed = feedparser.parse(response.text)

        if not feed.entries:
            break

        for entry in feed.entries:
            # Handle potential variation in entry ID format cleanly
            raw_id = entry.id
            if "/abs/" in raw_id:
                paper_id = raw_id.split("/abs/")[1]
            else:
                paper_id = raw_id.strip()

            title = entry.title.replace("\n", " ").strip()
            abstract = entry.summary.replace("\n", " ").strip()
            authors = [author.name for author in entry.authors]
            published = entry.published

            pdf_url = None
            for link in entry.links:
                if link.get("type") == "application/pdf":
                    pdf_url = link.href
                    break
            if not pdf_url:
                pdf_url = f"http://arxiv.org/pdf/{paper_id}"

            categories = [tag.term for tag in entry.tags]

            papers.append({
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
                "pdf_url": pdf_url,
                "categories": categories
            })

        fetched += len(feed.entries)
        time.sleep(ARXIV_RATE_LIMIT)

    return papers[:max_papers]


def download_pdfs(papers: list[dict], output_dir: str) -> list[dict]:
    """
    Download PDFs for each paper.

    Enforces 3-second delay between downloads.
    Skips papers whose PDF already exists on disk.
    Adds 'pdf_path' key to each paper dict.
    Logs and skips on download failure (do not raise).

    Args:
        papers: list of paper dicts from fetch_papers
        output_dir: path to data/pdfs/

    Returns: the same list with 'pdf_path' added to each dict
    """
    os.makedirs(output_dir, exist_ok=True)

    total = len(papers)
    downloaded = 0
    cached = 0
    failed = 0
    for i, paper in enumerate(papers, start=1):
        filename = paper["paper_id"].replace("/", "_") + ".pdf"
        pdf_path = os.path.join(output_dir, filename)

        if os.path.exists(pdf_path):
            paper["pdf_path"] = pdf_path
            cached += 1
        else:
            try:
                response = requests.get(paper["pdf_url"], timeout=30)
                response.raise_for_status()

                with open(pdf_path, "wb") as f:
                    f.write(response.content)

                paper["pdf_path"] = pdf_path
                downloaded += 1
            except Exception as e:
                failed += 1
                print(f"  [download] FAILED {paper['paper_id']}: {e}")

            time.sleep(ARXIV_RATE_LIMIT)

        if i % 25 == 0 or i == total:
            print(f"  [download] {i}/{total}  (new={downloaded} cached={cached} failed={failed})")

    return papers