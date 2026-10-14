from agvrag.ingestion.cleaning import (
    dehyphenate,
    find_repeated_margin_lines,
    garbled_score,
    is_noise_line,
    normalize_chars,
    reflow_paragraphs,
)


def test_ligatures_and_soft_hyphen():
    assert normalize_chars("e\ufb03cient \ufb01ne-tuning") == "efficient fine-tuning"
    assert normalize_chars("adapta\u00adtion") == "adaptation"


def test_dehyphenate_joins_line_breaks_but_keeps_compounds():
    assert dehyphenate("adapta-\ntion") == "adaptation"
    assert dehyphenate("BERT-\nLarge") == "BERT-Large"


def test_noise_lines():
    assert is_noise_line("12")
    assert is_noise_line("arXiv:2106.09685v2 [cs.CL] 16 Oct 2021")
    assert not is_noise_line("1 Introduction")


def test_repeated_margin_lines_detected():
    pages = [
        ["Published as a conference paper at ICLR 2022", f"body {i}", "more", str(i)]
        for i in range(6)
    ]
    rep = find_repeated_margin_lines(pages)
    assert (
        "Published as a conference paper at ICLR ####" in rep
        or "Published as a conference paper at ICLR #" in rep
    )


def test_reflow_paragraphs():
    assert (
        reflow_paragraphs("one line\nwrapped here\n\nnext para")
        == "one line wrapped here\n\nnext para"
    )


def test_garbled_score_separates_clean_from_garbage():
    clean = "Retrieval augmented generation improves factual accuracy on open domain tasks. " * 5
    junk = "\u2211\u222b\u2202 xqzv bcdfg \u2297\u2295 " * 20
    assert garbled_score(clean) < 0.1
    assert garbled_score(junk) > 0.5
