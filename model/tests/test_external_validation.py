from pathlib import Path

from arkansas_pharma_signal.external_validation import (
    evaluate_band_reference, evaluate_biocaster_reference,
    evaluate_cirad_reference, evaluate_daniel_reference, html_text,
)


def test_external_html_parser_excludes_script_text(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html><script>ignore()</script><body>Outbreak&nbsp; report</body></html>")
    assert html_text(path) == "Outbreak report"


def test_cirad_reference_is_evaluation_only_and_provenanced():
    result = evaluate_cirad_reference(Path(__file__).parents[2])
    assert result["available"] is True
    assert result["sentence_count"] == 1244
    assert result["article_count"] == 88
    assert result["training_use"] == "evaluation_only"
    assert 0.0 <= result["accuracy"] <= 1.0


def test_biocaster_reference_is_recall_only_and_provenanced():
    result = evaluate_biocaster_reference(Path(__file__).parents[2])
    assert result["available"] is True
    assert result["annotated_event_frame_count"] == 200
    assert result["evaluated_article_pages"] >= 1
    assert result["evaluation_scope"].endswith("recall only")
    assert result["training_use"] == "evaluation_only"


def test_band_reference_is_recall_only_and_provenanced():
    result = evaluate_band_reference(Path(__file__).parents[2])
    assert result["available"] is True
    assert result["ner_test_documents"] >= 1000
    assert result["unique_outbreak_contexts"] >= 100
    assert result["evaluation_scope"].endswith("recall only")
    assert result["training_use"] == "evaluation_only"


def test_daniel_reference_is_entity_recall_only_and_provenanced():
    result = evaluate_daniel_reference(Path(__file__).parents[2])
    assert result["available"] is True
    assert result["disease_span_count"] >= 20
    assert result["evaluation_scope"].endswith("recall only")
    assert result["training_use"] == "evaluation_only"
