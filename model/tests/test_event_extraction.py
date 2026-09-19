import pandas as pd

from arkansas_pharma_signal.event_extraction import extract_events


def test_event_extraction_preserves_span_and_time():
    corpus = pd.DataFrame([{
        "article_id": "a1", "title": "Drug shortage reported",
        "body": "Officials said a possible shortage may affect patients.",
        "published_at": "2024-01-02T00:00:00Z", "source": "local",
        "city": "Little Rock", "is_full_text": True,
    }])
    events = extract_events(corpus, ["metformin"])
    assert len(events) == 1
    row = events.iloc[0]
    assert row["article_id"] == "a1"
    assert "shortage" in row["evidence_span"].lower()
    assert row["uncertainty_status"] == "uncertain"
    assert row["source_timestamp"] == "2024-01-02T00:00:00Z"


if __name__ == "__main__":
    test_event_extraction_preserves_span_and_time()
    print("ok")
