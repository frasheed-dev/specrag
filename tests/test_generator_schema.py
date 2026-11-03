from specrag.generator import SpecAnswer


def test_spec_answer_validation():
    data = {
        "answer": "The H100 NVL has 94 GB memory per GPU.",
        "cited_clause_ids": ["Table 2. Memory Specifications"],
        "verbatim_evidence": ["| Memory size | 94 GB |"],
        "confidence": 0.98,
        "is_answerable": True,
    }

    ans = SpecAnswer(**data)
    assert ans.is_answerable is True
    assert ans.confidence == 0.98
    assert "Table 2. Memory Specifications" in ans.cited_clause_ids
    assert len(ans.verbatim_evidence) == 1


def test_spec_answer_defaults():
    ans = SpecAnswer(answer="Direct statement.")
    assert ans.confidence == 1.0
    assert ans.is_answerable is True
    assert ans.cited_clause_ids == []
    assert ans.verbatim_evidence == []
