import pytest
import requests

from biastracker.annotations.panther import (
    fetch_panther_annotations,
    fetch_panther_geneinfo_batch,
    load_panther_annotation,
    load_panther_api_annotations,
    parse_panther_geneinfo,
    run_panther_overrep,
)


class DummyResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)

def test_panther_loader_inference(tmp_path):
    p = tmp_path / "fake_panther.csv"
    p.write_text("uniprot,go_id,name\nP12345,GO:0001,Some process\nP67890,GO:0002,Other process\n")
    
    ann = load_panther_annotation(p, name="test_panther", panther_type="go_slim")
    assert ann.name == "test_panther"
    assert ann.source == "PANTHER"
    assert "P12345" in ann.ids_for_term("Some process")
    assert len(ann.table) == 2
    assert ann.table["category"].iloc[0] == "ontology"
    assert ann.table["primary_id"].iloc[0] == "P12345"
    assert ann.table["term_id"].iloc[0] == "GO:0001"
    assert ann.table["term_name"].iloc[0] == "Some process"

def test_panther_loader_missing_cols(tmp_path):
    p = tmp_path / "fake_panther2.csv"
    p.write_text("random_id,random_term,random_name\nP12345,GO:0001,Some process\n")
    
    with pytest.raises(ValueError, match="Could not infer or find required columns"):
        load_panther_annotation(p, name="test2", panther_type="family")
        
def test_panther_categories(tmp_path):
    p = tmp_path / "fake.csv"
    p.write_text("uniprot,go_id,name\nP1,G1,N1\n")
    
    ann1 = load_panther_annotation(p, name="t1", panther_type="family")
    assert ann1.table["category"].iloc[0] == "family"
    
    ann2 = load_panther_annotation(p, name="t2", panther_type="reactome")
    assert ann2.table["category"].iloc[0] == "pathway"
    
    ann3 = load_panther_annotation(p, name="t3", panther_type="protein_class")
    assert ann3.table["category"].iloc[0] == "protein_class"
    
    ann4 = load_panther_annotation(p, name="t4", panther_type="unknown_type")
    assert ann4.table["category"].iloc[0] == "unknown"

def test_panther_explicit_cols_and_category(tmp_path):
    p = tmp_path / "fake_panther3.csv"
    p.write_text("random_id,random_term,random_name\nP12345,GO:0001,Some process\n")
    
    ann = load_panther_annotation(
        p, 
        name="test3", 
        panther_type="family",
        id_col="random_id",
        term_id_col="random_term",
        term_name_col="random_name",
        category="custom_category"
    )
    
    assert ann.table["category"].iloc[0] == "custom_category"
    assert ann.table["primary_id"].iloc[0] == "P12345"
    assert ann.table["term_id"].iloc[0] == "GO:0001"
    assert ann.table["term_name"].iloc[0] == "Some process"


def test_parse_panther_geneinfo_handles_single_dict_shapes():
    payload = {
        "search": {
            "mapped_genes": {
                "gene": {
                    "input_id": "P12345",
                    "accession": "PTHR10000:SF1",
                    "annotation_type_list": {
                        "annotation_data_type": {
                            "content": "GO:0008150",
                            "annotation_list": {
                                "annotation": {
                                    "id": "GO:0009987",
                                    "name": "cellular process",
                                }
                            },
                        }
                    },
                }
            }
        }
    }

    df = parse_panther_geneinfo(payload, requested_ids=["P12345"])

    assert len(df) == 1
    assert df.loc[0, "primary_id"] == "P12345"
    assert df.loc[0, "term_id"] == "GO:0009987"
    assert df.loc[0, "term_name"] == "cellular process"
    assert df.loc[0, "category"] == "go_bp"
    assert df.loc[0, "panther_accession"] == "PTHR10000:SF1"
    assert df.loc[0, "panther_dataset_id"] == "GO:0008150"


def test_parse_panther_geneinfo_handles_list_shapes_and_requested_id_mapping():
    """Genes are identified via UniProtKB= embedded in the panther_accession string."""
    payload = {
        "search": {
            "mapped_genes": {
                "gene": [
                    {
                        # No input_id — primary_id resolved from UniProtKB in accession.
                        "accession": "HUMAN|HGNC=11111|UniProtKB=P11111",
                        "annotation_type_list": {
                            "annotation_data_type": [
                                {
                                    "content": "GO:0005575",
                                    "annotation_list": {
                                        "annotation": [
                                            {"id": "GO:0005737", "label": "cytoplasm"},
                                            {"id": "GO:0005829", "name": "cytosol"},
                                        ]
                                    },
                                },
                                {
                                    "content": "ANNOT_TYPE_ID_PANTHER_PC",
                                    "annotation_list": {
                                        "annotation": {
                                            "id": "PC00001",
                                            "name": "enzyme modulator",
                                        }
                                    },
                                },
                            ]
                        },
                    },
                    {
                        "accession": "HUMAN|HGNC=22222|UniProtKB=Q22222",
                        "annotation_type_list": {
                            "annotation_data_type": {
                                "content": "ANNOT_TYPE_ID_PANTHER_PATHWAY",
                                "annotation_list": {
                                    "annotation": {
                                        "id": "P00001",
                                        "name": "Some pathway",
                                    }
                                },
                            }
                        },
                    },
                ]
            }
        }
    }

    df = parse_panther_geneinfo(payload, requested_ids=["P11111", "Q22222"])

    assert set(df["primary_id"]) == {"P11111", "Q22222"}
    assert set(df["category"]) == {"go_cc", "protein_class", "panther_pathway"}
    assert len(df) == 4



def test_parse_panther_geneinfo_filters_categories():
    payload = {
        "search": {
            "mapped_genes": {
                "gene": {
                    "input_id": "P12345",
                    "annotation_type_list": {
                        "annotation_data_type": [
                            {
                                "content": "GO:0008150",
                                "annotation_list": {
                                    "annotation": {"id": "GO:1", "name": "process"}
                                },
                            },
                            {
                                "content": "GO:0003674",
                                "annotation_list": {
                                    "annotation": {"id": "GO:2", "name": "binding"}
                                },
                            },
                        ]
                    },
                }
            }
        }
    }

    df = parse_panther_geneinfo(payload, categories=["go_mf"])

    assert len(df) == 1
    assert df.loc[0, "category"] == "go_mf"
    assert df.loc[0, "term_name"] == "binding"


def test_parse_panther_geneinfo_empty_annotations_returns_expected_columns():
    payload = {
        "search": {
            "mapped_genes": {
                "gene": {
                    "input_id": "P12345",
                    "annotation_type_list": {"annotation_data_type": []},
                }
            }
        }
    }

    df = parse_panther_geneinfo(payload, requested_ids=["P12345"])

    assert df.empty
    assert list(df.columns) == [
        "primary_id",
        "term_id",
        "term_name",
        "source",
        "category",
        "panther_accession",
        "panther_dataset_id",
    ]


def test_fetch_panther_geneinfo_batch_posts_form_and_raises_api_error():
    session = DummySession([DummyResponse({"search": {"error": "bad ids"}})])

    with pytest.raises(RuntimeError, match="bad ids"):
        fetch_panther_geneinfo_batch(["P12345"], organism=9606, session=session)

    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url.endswith("/geneinfo")
    assert kwargs["data"] == {"organism": "9606", "geneInputList": "P12345"}
    assert kwargs["headers"] == {"Accept": "application/json"}


def test_fetch_panther_geneinfo_batch_requires_ids():
    with pytest.raises(ValueError, match="No IDs provided"):
        fetch_panther_geneinfo_batch([])


def test_fetch_panther_annotations_uses_cache(tmp_path):
    payload = {
        "search": {
            "mapped_genes": {
                "gene": {
                    "input_id": "P12345",
                    "annotation_type_list": {
                        "annotation_data_type": {
                            "content": "GO:0008150",
                            "annotation_list": {
                                "annotation": {"id": "GO:1", "name": "process"}
                            },
                        }
                    },
                }
            }
        }
    }
    first_session = DummySession([DummyResponse(payload)])
    first = fetch_panther_annotations(
        ["P12345"],
        categories=["go_bp"],
        cache_dir=tmp_path,
        session=first_session,
    )
    second_session = DummySession([])
    second = fetch_panther_annotations(
        ["P12345"],
        categories=["go_bp"],
        cache_dir=tmp_path,
        session=second_session,
    )

    assert len(first) == 1
    assert second.equals(first)
    assert second_session.calls == []


def test_load_panther_api_annotations_returns_annotationset(tmp_path):
    payload = {
        "search": {
            "mapped_genes": {
                "gene": {
                    "input_id": "P12345",
                    "annotation_type_list": {
                        "annotation_data_type": {
                            "content": "GO:0003674",
                            "annotation_list": {
                                "annotation": {"id": "GO:0005515", "name": "binding"}
                            },
                        }
                    },
                }
            }
        }
    }
    session = DummySession([DummyResponse(payload)])

    # Prime the cache because the public loader intentionally keeps the simple
    # signature and lets fetch_panther_annotations own HTTP session details.
    fetch_panther_annotations(
        ["P12345"],
        categories=["go_mf"],
        cache_dir=tmp_path,
        session=session,
    )
    ann = load_panther_api_annotations(
        ["P12345"],
        name="api",
        categories=["go_mf"],
        cache_dir=tmp_path,
    )

    assert ann.name == "api"
    assert ann.source == "PANTHER"
    assert ann.ids_for_term("binding") == {"P12345"}
    assert ann.metadata["source"] == "PANTHER_API"
    assert ann.metadata["n_input_ids"] == 1


def test_run_panther_overrep_parses_successful_response_with_background():
    payload = {
        "results": {
            "result": [
                {
                    "term": {"id": "GO:0009987", "label": "cellular process"},
                    "expected": "1.5",
                    "fold_enrichment": "2.0",
                    "pValue": "0.01",
                    "fdr": "0.02",
                },
                {
                    "term": {"id": "GO:0008152", "label": "metabolic process"},
                    "expected": 4,
                    "foldEnrichment": 0.5,
                    "p_value": 0.04,
                    "FDR": 0.08,
                },
            ]
        }
    }
    session = DummySession([DummyResponse(payload)])

    df = run_panther_overrep(
        ["P12345", "Q67890", "P12345"],
        ref_ids=["A1", "A2"],
        annot_dataset="GO:0008150",
        session=session,
    )

    assert list(df.columns) == [
        "term_id",
        "term_name",
        "source",
        "category",
        "expected",
        "fold_enrichment",
        "p_value",
        "fdr",
        "direction",
    ]
    assert list(df["term_id"]) == ["GO:0009987", "GO:0008152"]
    assert list(df["category"]) == ["go_bp", "go_bp"]
    assert list(df["direction"]) == ["enriched", "depleted"]
    assert df.loc[0, "expected"] == 1.5
    assert df.loc[0, "p_value"] == 0.01
    assert df.loc[0, "fdr"] == 0.02

    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url.endswith("/enrich/overrep")
    assert kwargs["data"] == {
        "geneInputList": "P12345,Q67890",
        "organism": "9606",
        "annotDataSet": "GO:0008150",
        "enrichmentTestType": "FISHER",
        "correction": "FDR",
        "refInputList": "A1,A2",
        "refOrganism": "9606",
    }


def test_run_panther_overrep_handles_single_result_object():
    payload = {
        "results": {
            "result": {
                "term": {"id": "GO:0005515", "name": "protein binding"},
                "expected": "2",
                "fold_enrichment": "1",
                "pValue": "0.5",
                "fdr": "0.9",
            }
        }
    }
    session = DummySession([DummyResponse(payload)])

    df = run_panther_overrep(
        ["P12345"],
        annot_dataset="GO:0003674",
        session=session,
    )

    assert len(df) == 1
    assert df.loc[0, "term_id"] == "GO:0005515"
    assert df.loc[0, "term_name"] == "protein binding"
    assert df.loc[0, "category"] == "go_mf"
    assert df.loc[0, "direction"] == "neutral"


def test_run_panther_overrep_uses_synthetic_id_for_missing_term_id():
    payload = {
        "results": {
            "result": {
                "term": {"label": "unclassified"},
                "expected": "0.2",
                "fold_enrichment": "3",
                "pValue": "0.03",
                "fdr": "0.05",
            }
        }
    }
    session = DummySession([DummyResponse(payload)])

    df = run_panther_overrep(["P12345"], session=session)

    assert df.loc[0, "term_id"] == "PANTHER:UNCLASSIFIED"
    assert df.loc[0, "term_name"] == "unclassified"
    assert df.loc[0, "direction"] == "enriched"


def test_run_panther_overrep_raises_on_error_payload():
    session = DummySession([DummyResponse({"results": {"error": "bad request"}})])

    with pytest.raises(RuntimeError, match="bad request"):
        run_panther_overrep(["P12345"], session=session)


def test_run_panther_overrep_requires_query_ids():
    with pytest.raises(ValueError, match="No query IDs provided"):
        run_panther_overrep([])
