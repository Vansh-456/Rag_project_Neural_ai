from agvrag.utils.text import count_tokens, jaccard, normalize_name, shingles, split_sentences


def test_count_tokens_is_deterministic_and_monotone():
    assert count_tokens("") == 0
    a = count_tokens("Low-rank adaptation of large language models.")
    assert a == count_tokens("Low-rank adaptation of large language models.")
    assert count_tokens("Low-rank adaptation of large language models. " * 2) > a


def test_count_tokens_tracks_bpe_scale():
    # ~0.75 words per token for English prose: 40 words should be ~45-65 tokens
    text = (
        "Retrieval augmented generation combines a parametric language model with a non "
        "parametric memory so that the model can condition on retrieved passages when it "
        "generates an answer, which improves factuality on knowledge intensive tasks and "
        "makes the source of each claim inspectable by a human reader."
    )
    n = count_tokens(text)
    assert 40 <= n <= 75


def test_sentence_split_keeps_abbreviations():
    s = split_sentences("We follow Hu et al. in this setup. Results are in Fig. 3 below. Done.")
    assert s == ["We follow Hu et al. in this setup.", "Results are in Fig. 3 below.", "Done."]


def test_normalize_name():
    assert normalize_name("Low–Rank  Adaptation") == "low rank adaptation"
    assert normalize_name("GPT-3") == "gpt 3"


def test_jaccard():
    a, b = shingles("the quick brown fox jumps"), shingles("the quick brown fox jumps")
    assert jaccard(a, b) == 1.0
    assert jaccard(a, shingles("completely different text here now")) == 0.0
