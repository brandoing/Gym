# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, patch

import pytest

from resources_servers.clinical_consultation.app import ClinicalCase, ClinicalConsultationServer
from resources_servers.clinical_consultation.tests.test_app import case, make_server, published_case, request


def released_case(*, unweighted=False):
    row = published_case()
    row.update(case_format="published_rubric_v1", scenario="published_clinical_task", reference_answer=None)
    row["provenance"].update(
        dataset="AQ-MedAI/ClinAlign" if unweighted else "Estrellajer/ProRubric",
        dataset_revision="a" * 40 if unweighted else "sha256:" + "b" * 64,
        source_split="public_demo" if unweighted else "train",
        source_file="release.jsonl",
        protocol_dimensions=[{"criterion": "PRIVATE_PROTOCOL"}],
    )
    if unweighted:
        for rubric in row["rubrics"]:
            rubric["points"] = None
    return row


@pytest.mark.parametrize("unweighted", [False, True])
async def test_missing_reference_and_source_weights_reach_only_judge(tmp_path, unweighted):
    row = released_case(unweighted=unweighted)
    server = make_server(tmp_path, cases=[row])
    with patch.object(ClinicalConsultationServer, "_judge", new_callable=AsyncMock) as judge:
        judge.return_value = {"score": 7, "rationale": "rubric satisfied"}
        result = await server.verify(request("candidate"))
    assert result.reward == pytest.approx(6 / 9)
    guide, context = judge.call_args.args[:2]
    assert "reference_answer" not in json.loads(context)
    assert "PRIVATE_PROTOCOL" not in guide + context
    assert "verbatim published criteria" in guide
    assert ('"points": null' in guide) is unweighted
    assert result.responses_create_params.input == request().responses_create_params.input
    assert all(key not in result.model_dump() for key in ("provenance", "rubrics", "reference_answer"))


@pytest.mark.parametrize("mode", ["additive", "protocol"])
def test_no_unqualified_published_reward_arm(tmp_path, mode):
    with pytest.raises(ValueError, match="require rar_holistic"):
        make_server(tmp_path, cases=[released_case(unweighted=True)], reward_mode=mode)


@pytest.mark.parametrize("factory", [case, published_case])
@pytest.mark.parametrize("change", ["reference", "weights"])
def test_new_format_does_not_weaken_existing_contract(factory, change):
    row = factory()
    if change == "reference":
        row["reference_answer"] = None
    else:
        row["rubrics"][0]["points"] = None
    with pytest.raises(ValueError):
        ClinicalCase.model_validate(row)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", "unqualified/source"),
        ("dataset_revision", "main"),
        ("source_row_sha256", "missing"),
        ("source_row_index", True),
        ("source_split", "test"),
    ],
)
def test_published_identity_and_holdouts_are_enforced(field, value):
    row = released_case()
    row["provenance"][field] = value
    with pytest.raises(ValueError):
        ClinicalCase.model_validate(row)


@pytest.mark.parametrize("unweighted", [False, True])
def test_no_invented_or_discarded_weights(unweighted):
    row = released_case(unweighted=unweighted)
    row["rubrics"][0]["points"] = 1 if unweighted else None
    with pytest.raises(ValueError):
        ClinicalCase.model_validate(row)
