# This file splits a long text into smaller overlapping chunks
# Smaller chunks = better search results (more focused)

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> list:
    """
    Splits text recursively by trying to split on:
    1. Paragraphs (\\n\\n)
    2. Lines (\\n)
    3. Words ( )
    This prevents cutting sentences or lists in the middle.
    
    Args:
        text: the full extracted text from the PDF
        chunk_size: maximum characters per chunk
        overlap: how many characters to overlap from previous chunk
        
    Returns:
        A list of text chunks (strings)
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", " ", ""]

    def _split(text_to_split: str, separators_list: list) -> list:
        if len(text_to_split) <= chunk_size:
            return [text_to_split] if text_to_split.strip() else []

        if not separators_list:
            # Force split if no separators left
            chunks = []
            start = 0
            while start < len(text_to_split):
                chunk = text_to_split[start:start+chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
                start += chunk_size - overlap
            return chunks

        sep = separators_list[0]
        if sep == "":
            parts = list(text_to_split)
        else:
            parts = text_to_split.split(sep)

        chunks = []
        current_chunk = []
        current_length = 0

        for part in parts:
            if len(part) > chunk_size:
                # Flush current
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                # Recursively split the oversized part
                sub_parts = _split(part, separators_list[1:])
                chunks.extend(sub_parts)
                continue

            # Check if this part fits in the current chunk
            sep_len = len(sep) if current_chunk else 0
            if current_length + sep_len + len(part) > chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_parts = []
                overlap_len = 0
                for prev_part in reversed(current_chunk):
                    prev_sep_len = len(sep) if overlap_parts else 0
                    if overlap_len + prev_sep_len + len(prev_part) <= overlap:
                        overlap_parts.insert(0, prev_part)
                        overlap_len += prev_sep_len + len(prev_part)
                    else:
                        break
                
                current_chunk = overlap_parts + [part]
                current_length = overlap_len + (len(sep) if overlap_parts else 0) + len(part)
            else:
                current_chunk.append(part)
                current_length += sep_len + len(part)

        if current_chunk:
            chunks.append(sep.join(current_chunk))

        return [c.strip() for c in chunks if c.strip()]

    return _split(text, separators)

