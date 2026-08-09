import re
import tiktoken

CHUNK_SIZE = 500
# Sentences carried from the end of one chunk into the start of the next, so a fact
# split across a boundary is retrievable from either side. Two sentences lands around
# 11% of a chunk; one sentence measured 5.8%, below the 10-15% target.
CHUNK_OVERLAP_SENTENCES = 2


def clean_inline_noise(text: str) -> str:
    """
    Remove standalone page numbers or floating layout headers,
    and clean up excessive consecutive newlines caused by layout gaps.
    """
    # Remove standalone page numbers or floating layout headers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Clean up excessive consecutive newlines/whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def exclude_garbage_sections(text: str) -> str:
    """
    Finds common markers for references or acknowledgments at the end 
    of academic papers and truncates the text to drop them.
    """
    # Look for standalone lines indicating the end of the content body
    garbage_pattern = r'(?im)^references\s*$|^bibliography\s*$|^acknowledgments\s*$'
    
    match = re.search(garbage_pattern, text)
    if match:
        # Cut the document off right before the garbage section starts
        return text[:match.start()].rstrip()
    
    return text


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, handling common academic abbreviations.

    Avoids false splits on:
    - Abbreviations: et al., Fig., vs., i.e., e.g., Dr., Prof., Inc., Ltd., etc.
    - Single initials: J. Smith, A. Vaswani
    - Numbered references: Table 3, Figure 2, etc.
    """
    # First clean up the inline layout noise
    text = clean_inline_noise(text)

    # Common abbreviations that shouldn't trigger sentence breaks
    abbreviations = (
        r'\bet\s+al\.|'  # et al.
        r'\bFig\.|'       # Fig.
        r'\bvs\.|'        # vs.
        r'\bi\.e\.|'      # i.e.
        r'\be\.g\.|'      # e.g.
        r'\b[Dd]r\.|'     # Dr.
        r'\b[Pp]rof\.|'   # Prof.
        r'\b[Mm]r\.|'     # Mr.
        r'\b[Mm]rs\.|'    # Mrs.
        r'\b[Mm]s\.|'     # Ms.
        r'\b[Ii]nc\.|'    # Inc.
        r'\b[Ll]td\.|'    # Ltd.
        r'\b[Cc]o\.|'     # Co.
        r'\b[Ss]r\.|'     # Sr.
        r'\b[Jj]r\.|'     # Jr.
        r'\b[Pp]\.\s*[0-9]|'  # p. 123
        r'\b[Pp]p\.\s*[0-9]|' # pp. 123
        r'\b[Vv]ol\.|'    # vol.
        r'\b[Nn]o\.|'     # no.
        r'\b[Ee]d\.|'     # ed.
        r'\b[Ee]ds\.|'    # eds.
        r'\b[Aa]pprox\.|' # approx.
        r'\b[Mm]in\.|'    # min.
        r'\b[Ss]ec\.|'    # sec.
        r'\b[Hh]ref|'     # href (URLs)
        r'\b[Ww]ww\.|'    # www.
        r'\.org|\.com|\.edu|\.net|\.io'  # domain extensions
    )

    # Protect abbreviations and single initials by replacing periods with placeholder
    placeholder_map = {}
    abbrev_pattern = abbreviations.rstrip('|')

    def protect_abbrev(match):
        text_match = match.group(0)
        placeholder = f"__ABBREV_{len(placeholder_map)}__"
        placeholder_map[placeholder] = text_match
        return placeholder

    protected_text = re.sub(abbrev_pattern, protect_abbrev, text)

    # Also protect single initials (e.g., "J. Smith", "A. Vaswani")
    protected_text = re.sub(r'\b([A-Z])\.\s+(?=[A-Z][a-z])', lambda m: f"__INITIAL_{m.group(1)}__", protected_text)

    # Also protect numbered references before capital words (e.g., "Table 3. Caption")
    protected_text = re.sub(
        r'\b(Table|Figure|Fig|Equation|Eq|Sec|Section|Chapter|Appendix|Algorithm)\s+(\d+)\.\s+(?=[A-Z])',
        lambda m: f"__NUMREF_{m.group(1).replace(' ', '_')}_{m.group(2)}__",
        protected_text,
        flags=re.IGNORECASE
    )

    # Split on sentence boundaries: . ! ? followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected_text)

    # Restore abbreviations, initials, and references
    restored_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Restore all protected patterns
            for placeholder, abbrev in placeholder_map.items():
                sentence = sentence.replace(placeholder, abbrev)
            # Restore initials
            sentence = re.sub(r'__INITIAL_([A-Z])__', r'\1.', sentence)
            # Restore numbered references
            sentence = re.sub(r'__NUMREF_([A-Za-z_]+)_(\d+)__', lambda m: f"{m.group(1).replace('_', ' ')} {m.group(2)}.", sentence)
            restored_sentences.append(sentence)

    return restored_sentences


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, tolerance_factor: float = 0.2) -> list[str]:
    """
    Split text into chunks at sentence boundaries.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    threshold = chunk_size * (1 + tolerance_factor)

    sentences = _split_sentences(text)
    chunks = []
    buffer = []
    buffer_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence, disallowed_special=()))

        if buffer and buffer_tokens + sentence_tokens > threshold:
            chunk_text_str = " ".join(buffer)
            if chunk_text_str.strip() and len(encoding.encode(chunk_text_str, disallowed_special=())) >= 20:
                chunks.append(chunk_text_str)
                overlap_count = min(CHUNK_OVERLAP_SENTENCES, len(buffer) - 1)
                overlap = buffer[-overlap_count:] if overlap_count else []
                overlap_tokens = sum(
                    len(encoding.encode(item, disallowed_special=()))
                    for item in overlap
                )
                if overlap_tokens + sentence_tokens <= threshold:
                    buffer = overlap
                    buffer_tokens = overlap_tokens
                else:
                    buffer = []
                    buffer_tokens = 0
            else:
                buffer = []
                buffer_tokens = 0

        buffer.append(sentence)
        buffer_tokens += sentence_tokens

    if buffer:
        chunk_text_str = " ".join(buffer)
        if chunk_text_str.strip() and len(encoding.encode(chunk_text_str, disallowed_special=())) >= 20:
            chunks.append(chunk_text_str)

    return chunks


def extract_abstract(text: str) -> tuple[str, int]:
    """
    Extract abstract section from paper text.

    Returns: (abstract_text, end_position) where end_position is the
    character index in the original text where the abstract ends.
    Returns ("", 0) if no abstract found.
    """
    encoding = tiktoken.get_encoding("cl100k_base")

    abstract_match = re.search(r'(?im)^abstract\s*$|^abstract[:\s]', text)
    if not abstract_match:
        return "", 0

    abstract_start_pos = abstract_match.start()
    remaining_text = text[abstract_start_pos:]

    section_match = re.search(
        r'(?im)^(1\.?\s+introduction|introduction\s*$)|\n\d+[\.\s]',
        remaining_text
    )

    if section_match:
        abstract_end_pos = abstract_start_pos + section_match.start()
        abstract_text = remaining_text[:section_match.start()]
    else:
        abstract_end_pos = len(text)
        abstract_text = remaining_text

    abstract_text = re.sub(r'(?im)^abstract[:\s]*', '', abstract_text).strip()

    if len(encoding.encode(abstract_text, disallowed_special=())) < 30:
        return "", 0

    return abstract_text, abstract_end_pos
