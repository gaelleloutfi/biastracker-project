import pandas as pd
import pytest
import requests

from biastracker.annotations.panther import (
    PANTHER_OUTPUT_COLUMNS,
    fetch_panther_annotations,
    fetch_panther_geneinfo_batch,
    parse_panther_geneinfo,
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


def test_parse_panther_api_handles_nested_dict_shapes():
    payload = {
        "search": {
            "mapped_genes": {
                "gene": {
                    "input_id": "P04637",
                    "accession": "PTHR11447:SF6",
                    "annotation_type_list": {
                        "annotation_data_type": {
                            "content": "GO:0008150",
                            "annotation_list": {
                                "annotation": {
                                    "id": "GO:0006915",
                                    "name": "apoptotic process",
                                }
                            },
                        }
                    },
                }
            }
        }
    }

    df = parse_panther_geneinfo(payload, requested_ids=["P04637"])

    assert list(df.columns) == PANTHER_OUTPUT_COLUMNS
    assert df.to_dict("records") == [
        {
            "primary_id": "P04637",
            "term_id": "GO:0006915",
            "term_name": "apoptotic process",
            "source": "PANTHER",
            "category": "go_bp",
            "panther_accession": "PTHR11447:SF6",
            "panther_dataset_id": "GO:0008150",
        }
    ]


def test_parse_panther_api_handles_gene_annotation_type_and_annotation_lists():
    """Genes identified via UniProtKB= in panther_accession are parsed correctly."""
    payload = {
        "search": {
            "mapped_genes": {
                "gene": [
                    {
                        # No input_id field — primary_id must come from UniProtKB in accession.
                        "accession": "HUMAN|HGNC=11998|UniProtKB=P04637",
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
                                            "id": "PC00021",
                                            "name": "transcription factor",
                                        }
                                    },
                                },
                            ]
                        },
                    },
                    {
                        "input_id": "P00533",
                        "accession": "HUMAN|HGNC=3236|UniProtKB=P00533",
                        "annotation_type_list": {
                            "annotation_data_type": {
                                "content": "ANNOT_TYPE_ID_PANTHER_PATHWAY",
                                "annotation_list": {
                                    "annotation": {"id": "P00018", "name": "EGF receptor signaling pathway"}
                                },
                            }
                        },
                    },
                ]
            }
        }
    }

    df = parse_panther_geneinfo(payload, requested_ids=["P04637", "P00533"])

    assert set(df["primary_id"]) == {"P04637", "P00533"}
    assert set(df["category"]) == {"go_cc", "protein_class", "panther_pathway"}
    assert len(df) == 4


def test_parse_panther_geneinfo_primary_id_from_uniprot_in_accession():
    """Regression: primary_id must come from UniProtKB= in panther_accession, not
    from the position of the gene in the response list.

    If panther_accession contains UniProtKB=P38398, the row must have
    primary_id == 'P38398', never the first entry of requested_ids.
    """
    payload = {
        "search": {
            "mapped_genes": {
                "gene": {
                    # No input_id — would previously trigger the index fallback
                    # and (wrongly) assign the first requested ID (P04637).
                    "accession": "HUMAN|HGNC=1100|UniProtKB=P38398",
                    "annotation_type_list": {
                        "annotation_data_type": {
                            "content": "GO:0008150",
                            "annotation_list": {
                                "annotation": {
                                    "id": "GO:0006281",
                                    "name": "DNA repair",
                                }
                            },
                        }
                    },
                }
            }
        }
    }

    df = parse_panther_geneinfo(payload, requested_ids=["P04637", "P38398"])

    assert len(df) == 1
    row = df.iloc[0]
    # Must be P38398 (from panther_accession), NOT P04637 (first requested ID).
    assert row["primary_id"] == "P38398", (
        f"primary_id should be 'P38398' but got '{row['primary_id']}'"
    )
    assert row["panther_accession"] == "HUMAN|HGNC=1100|UniProtKB=P38398"
    assert row["term_id"] == "GO:0006281"


def test_parse_panther_api_category_filtering():
    payload = {
        "search": {
            "mapped_genes": {
                "gene": {
                    "input_id": "Q00987",
                    "annotation_type_list": {
                        "annotation_data_type": [
                            {
                                "content": "GO:0008150",
                                "annotation_list": {"annotation": {"id": "GO:1", "name": "process"}},
                            },
                            {
                                "content": "GO:0003674",
                                "annotation_list": {"annotation": {"id": "GO:2", "name": "binding"}},
                            },
                        ]
                    },
                }
            }
        }
    }

    df = parse_panther_geneinfo(payload, categories=["go_mf"])

    assert len(df) == 1
    assert df.iloc[0]["category"] == "go_mf"
    assert df.iloc[0]["term_name"] == "binding"


def test_parse_panther_api_empty_annotations_returns_expected_columns():
    df = parse_panther_geneinfo(
        {
            "search": {
                "mapped_genes": {
                    "gene": {
                        "input_id": "P04637",
                        "annotation_type_list": {"annotation_data_type": []},
                    }
                }
            }
        },
        requested_ids=["P04637"],
    )

    assert df.empty
    assert list(df.columns) == PANTHER_OUTPUT_COLUMNS


def test_panther_api_error_payload_raises_clear_error():
    session = DummySession([DummyResponse({"search": {"error": "bad input"}})])

    with pytest.raises(RuntimeError, match="PANTHER geneinfo error: bad input"):
        fetch_panther_geneinfo_batch(["P04637"], session=session)


def test_fetch_panther_api_annotations_cache_and_empty_shape(tmp_path):
    session = DummySession(
        [
            DummyResponse(
                {
                    "search": {
                        "mapped_genes": {
                            "gene": {
                                "input_id": "P04637",
                                "annotation_type_list": {"annotation_data_type": []},
                            }
                        }
                    }
                }
            )
        ]
    )

    first = fetch_panther_annotations(
        ["P04637"],
        categories=["go_bp"],
        cache_dir=tmp_path,
        session=session,
    )
    second = fetch_panther_annotations(
        ["P04637"],
        categories=["go_bp"],
        cache_dir=tmp_path,
        session=DummySession([]),
    )

    assert isinstance(first, pd.DataFrame)
    assert first.empty
    assert list(first.columns) == PANTHER_OUTPUT_COLUMNS
    assert second.empty
    assert list(second.columns) == PANTHER_OUTPUT_COLUMNS
