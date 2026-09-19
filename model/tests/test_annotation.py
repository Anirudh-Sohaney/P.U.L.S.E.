import pandas as pd

from arkansas_pharma_signal.annotation import ANNOTATION_COLUMNS, build_annotation_manifest


def test_annotation_manifest_has_no_generated_labels():
    corpus = pd.DataFrame([{"article_id": "a", "published_at": "2024-01-01",
                            "source": "s", "title": "shortage", "body": "a shortage",
                            "is_full_text": True}])
    out = build_annotation_manifest(corpus, n=2)
    assert list(out.columns) == ANNOTATION_COLUMNS
    assert out.iloc[0]["label_status"] == "unlabeled"
    assert out.iloc[0]["event_type_gold"] == ""


if __name__ == "__main__":
    test_annotation_manifest_has_no_generated_labels()
    print("ok")
