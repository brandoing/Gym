# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymChatCompletion
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.clinical_consultation.app import (
    CLINICAL_SCENARIOS,
    VERIFIER_FIXTURE,
    ClinicalCase,
    ClinicalConsultationConfig,
    ClinicalConsultationServer,
    ClinicalVerifyRequest,
    load_case_bank,
)


async def test_verifier_onboarding_contract():
    results = await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=(0, 1), determinism="stochastic")
    assert [result.kind for result in results] == ["full_reward", "zero_reward", "malformed"]


def case():
    return {
        "case_id": "fixture-1",
        "family_id": "family-1",
        "split": "train",
        "scenario": "simulation",
        "messages": [{"role": "user", "content": "Fictional simulation: red signal means stop. The signal is red."}],
        "reference_answer": "Stop the simulation.",
        "rubrics": [
            {"id": "stop", "criterion": "Stops the simulation", "points": 2},
            {"id": "unsafe", "criterion": "Continues the simulation", "points": -3},
        ],
        "protocols": [{"id": "safe", "criteria": ["stop"], "failure_criteria": ["unsafe"]}],
        "patient": {
            "initial_observation": "Signal visible",
            "facts": {"color": "red"},
            "required_facts": [],
            "expected_urgency": "stop",
            "reference_plan": "Stop the simulation.",
        },
        "provenance": {
            "source_id": "fictional",
            "source_kind": "fictional_engineering_fixture",
            "approved_for_training": False,
        },
    }


def clinical_case(scenario="documentation", *, user_type="clinician"):
    """A source-contract test record; no actual clinical generation is claimed."""
    bucket = {"documentation": "documentation", "evidence_research": "research"}.get(scenario, "consult")
    rubrics = [
        {"id": "faithful", "criterion": "Distinguishes recorded facts from missing details", "points": 5},
        {"id": "clear", "criterion": "Uses clear wording", "points": 2},
        {
            "id": "invented",
            "criterion": "Invents a detail absent from the source",
            "points": -5,
            "tags": ["unsupported_claim"],
        },
        {
            "id": "mistake",
            "criterion": "Assumes an unrecorded detail is normal",
            "points": -5,
            "tags": ["tempting_mistake"],
        },
        {"id": "padding", "criterion": "Adds irrelevant material", "points": -2, "tags": ["padding"]},
    ]
    if scenario == "documentation":
        rubrics[2]["tags"].append("invented_fact")
        rubrics.append(
            {"id": "omitted", "criterion": "Omits an essential source fact", "points": -4, "tags": ["omitted_fact"]}
        )
    if scenario == "urgent_escalation":
        rubrics.append(
            {
                "id": "delay",
                "criterion": "Buries required emergency action after the first sentence",
                "points": -6,
                "tags": ["delayed_escalation"],
            }
        )
    if scenario == "context_seeking":
        rubrics[4]["tags"].append("irrelevant_question")
    return case() | {
        "scenario": scenario,
        "benchmark_bucket": bucket,
        "user_type": user_type,
        "messages": [{"role": "user", "content": "Summarize only the facts in the supplied source excerpt."}],
        "reference_answer": "The supplied source does not record the missing detail.",
        "rubrics": rubrics,
        "key_decision_criterion_id": "faithful",
        "plausible_wrong_answer": "The source records an invented detail.",
        "contrast_failure_criterion_ids": ["invented"],
        "protocols": [],
        "patient": None,
        "provenance": {"source_id": "unit-test-source", "source_kind": "unit_test", "approved_for_training": False},
    }


def make_server(tmp_path, *, cases=None, **overrides):
    bank = tmp_path / "cases.jsonl"
    bank.write_text("".join(json.dumps(row) + "\n" for row in (cases or [case()])))
    config = ClinicalConsultationConfig(
        host="127.0.0.1",
        port=11000,
        entrypoint="app.py",
        case_bank_path=str(bank),
        case_bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
        judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        judge_chat_completion_params={"messages": [], "temperature": 0, "max_tokens": 1024},
        **overrides,
    )
    return ClinicalConsultationServer(config=config, server_client=MagicMock(spec=ServerClient))


def request(text="Stop the simulation.", **overrides):
    return ClinicalVerifyRequest.model_validate(
        {
            "case_id": "fixture-1",
            "responses_create_params": {"input": case()["messages"]},
            "response": {
                "id": "response-1",
                "created_at": 0,
                "model": "fixture",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": "message-1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            },
            **overrides,
        }
    )


def completion(content):
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "judge-1",
            "object": "chat.completion",
            "created": 0,
            "model": "scripted-judge",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
    )


@pytest.mark.parametrize("mode", ["additive", "protocol"])
async def test_rubric_verdicts_and_accounting(tmp_path, mode):
    server = make_server(tmp_path, reward_mode=mode, judge_model_revision="fixture-rev-1")
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(
            side_effect=[
                completion('{"criteria_met": true, "rationale": "Appropriate stop."}'),
                completion('{"criteria_met": false, "rationale": "Does not continue."}'),
            ]
        ),
    ) as judge:
        result = await server.verify(request())
    assert result.raw_score == 1
    assert result.reward == 1
    assert result.judge_calls == (2 if mode == "additive" else 1)
    assert not result.mask_sample
    assert result.judge_records[0]["usage"]["total_tokens"] == 14
    assert result.judge_records[0]["judge_model_revision"] == "fixture-rev-1"
    params = judge.call_args_list[0].kwargs["json"]
    assert "Stop the simulation." in params.messages[1]["content"]
    assert "Stops the simulation" in params.messages[0]["content"]
    assert all('"score"' not in call.kwargs["json"].messages[0]["content"] for call in judge.call_args_list)
    assert all(isinstance(record["verdict"]["criteria_met"], bool) for record in result.judge_records)


def test_plain_likert_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="reward_mode"):
        make_server(tmp_path, reward_mode="likert")


@pytest.mark.parametrize("score, reward", [(1, 0), (5.5, 0.5), (10, 1)])
async def test_holistic_default_normalizes_one_case_rubric_judgement(tmp_path, score, reward):
    server = make_server(tmp_path, judge_model_revision="fixture-rev-1")
    assert server.config.reward_mode == "rar_holistic"
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(return_value=completion(json.dumps({"score": score, "rationale": "Case rubric applied."}))),
    ) as judge:
        result = await server.verify(request())
    assert result.raw_score == score and result.reward == reward
    assert result.evaluation_completed and not result.mask_sample and result.judge_calls == 1
    params = judge.call_args.kwargs["json"]
    assert "Stops the simulation" in params.messages[0]["content"]
    assert "Continues the simulation" in params.messages[0]["content"]
    assert '"points": -3.0' in params.messages[0]["content"]
    context = json.loads(params.messages[1]["content"])
    assert context == {
        "conversation": case()["messages"],
        "reference_answer": case()["reference_answer"],
        "candidate": request().response.output_text,
    }
    record = result.judge_records[0]
    assert record["check_id"] == "overall" and record["verdict"]["score"] == score
    assert record["judge_model_revision"] == "fixture-rev-1"
    assert record["usage"]["total_tokens"] == 14


@pytest.mark.parametrize(
    "verdict",
    [
        "not JSON",
        "[]",
        "{}",
        '{"score":true}',
        '{"score":"10"}',
        '{"score":0}',
        '{"score":11}',
        '{"score":NaN}',
        '{"score":Infinity}',
        '{"criteria_met":true}',
    ],
)
async def test_invalid_holistic_judgement_is_masked_after_bounded_retries(tmp_path, verdict):
    server = make_server(tmp_path, judge_max_attempts=2)
    with patch("resources_servers.clinical_consultation.app.call_judge", AsyncMock(return_value=completion(verdict))):
        result = await server.verify(request())
    assert result.mask_sample and not result.evaluation_completed and result.judge_calls == 2
    assert result.raw_score is None
    assert all(record["status"] == "failed" for record in result.judge_records)
    assert result.model_dump()["instance_config"]["mask_sample"]


@pytest.mark.parametrize("mode", ["rar_holistic", "additive"])
async def test_each_prompt_uses_its_own_rubric(tmp_path, mode):
    first = case()
    second = case() | {
        "case_id": "fixture-2",
        "family_id": "family-2",
        "rubrics": [{"id": "difference", "criterion": "Explains the absolute event difference", "points": 2}],
        "protocols": [],
    }
    server = make_server(tmp_path, cases=[first, second], reward_mode=mode)
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(return_value=completion('{"score":10}' if mode == "rar_holistic" else '{"criteria_met":true}')),
    ) as judge:
        result = await server.verify(request(case_id="fixture-2"))
    assert result.reward == 1 and result.evaluation_completed and not result.mask_sample
    assert judge.call_count == 1
    prompt = judge.call_args.kwargs["json"].messages[0]["content"]
    assert "Explains the absolute event difference" in prompt and "Stops the simulation" not in prompt


async def test_request_cannot_override_private_bank(tmp_path):
    server = make_server(tmp_path)
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(return_value=completion('{"score":10}')),
    ) as judge:
        result = await server.verify(request(reference_answer="LEAK", rubrics=[{"criterion": "LEAK"}]))
    assert "LEAK" not in json.dumps(judge.call_args.kwargs["json"].model_dump())
    assert "reference_answer" not in result.model_dump()
    assert "rubrics" not in result.model_dump()
    assert result.reward == 1 and not result.mask_sample


@pytest.mark.parametrize("scenario", CLINICAL_SCENARIOS)
@pytest.mark.parametrize("user_type", ["clinician", "patient"])
async def test_tagged_tasks_use_private_case_rubric_without_patient_episode(tmp_path, scenario, user_type):
    row = clinical_case(scenario, user_type=user_type)
    server = make_server(tmp_path, cases=[row])
    assert server.get_case("fixture-1").patient is None
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(return_value=completion('{"score":8,"rationale":"Uses source faithfully."}')),
    ) as judge:
        result = await server.verify(
            request(
                "The supplied source does not record the missing detail.",
                scenario="FORGED_SCENARIO",
                benchmark_bucket="FORGED_BUCKET",
                user_type="FORGED_AUDIENCE",
                key_decision_criterion_id="FORGED_DECISION",
                plausible_wrong_answer="FORGED_CONTRAST",
                contrast_failure_criterion_ids=["FORGED_FAILURE"],
            )
        )
    params = judge.call_args.kwargs["json"]
    context = json.loads(params.messages[1]["content"])
    assert context["scenario"] == scenario
    assert context["benchmark_bucket"] == row["benchmark_bucket"]
    assert context["user_type"] == user_type
    assert context["reference_answer"] == row["reference_answer"]
    assert context["key_decision_criterion_id"] == "faithful"
    assert "plausible_wrong_answer" not in context and "contrast_failure_criterion_ids" not in context
    assert "Distinguishes recorded facts" in params.messages[0]["content"]
    assert "FORGED_" not in json.dumps(params.model_dump())
    assert (
        not {
            "scenario",
            "benchmark_bucket",
            "user_type",
            "patient",
            "key_decision_criterion_id",
            "plausible_wrong_answer",
            "contrast_failure_criterion_ids",
        }
        & result.model_dump().keys()
    )
    assert result.reward == pytest.approx(7 / 9) and result.judge_calls == 1 and not result.mask_sample


@pytest.mark.parametrize(
    "change",
    [
        {"scenario": "missing_information"},
        {"scenario": "evidence_interpretation"},
        {"scenario": "unknown"},
        {"benchmark_bucket": None},
        {"benchmark_bucket": "unknown"},
        {"user_type": None},
        {"user_type": "administrator"},
        {"benchmark_bucket": "consult"},
        {"scenario": "evidence_research", "benchmark_bucket": "consult"},
    ],
)
def test_bad_real_task_tags_fail_before_judging(tmp_path, change):
    with pytest.raises(ValueError):
        make_server(tmp_path, cases=[clinical_case() | change])


@pytest.mark.parametrize("missing", ["benchmark_bucket", "user_type"])
def test_real_task_tags_are_required(missing):
    row = clinical_case()
    del row[missing]
    with pytest.raises(ValueError, match="require benchmark_bucket and user_type"):
        ClinicalCase.model_validate(row)


def test_omitted_patient_is_valid_for_response_only_case():
    row = clinical_case()
    del row["patient"]
    assert ClinicalCase.model_validate(row).patient is None


@pytest.mark.parametrize("scenario", ["missing_information", "evidence_interpretation", "simulation"])
def test_legacy_fixture_labels_require_explicit_wiring_provenance(scenario):
    row = case() | {"scenario": scenario}
    assert ClinicalCase.model_validate(row).scenario == scenario
    row["provenance"].pop("source_kind")
    with pytest.raises(ValueError, match="legacy labels"):
        ClinicalCase.model_validate(row)


def test_wiring_fixture_cannot_claim_training_approval():
    row = case()
    row["provenance"]["approved_for_training"] = True
    with pytest.raises(ValueError, match="cannot be approved"):
        ClinicalCase.model_validate(row)


@pytest.mark.parametrize("key", [None, "unknown", "invented", "clear"])
def test_key_decision_must_be_a_highest_weight_positive_criterion(key):
    with pytest.raises(ValueError, match="highest-weight positive"):
        ClinicalCase.model_validate(clinical_case() | {"key_decision_criterion_id": key})


@pytest.mark.parametrize("count", [3, 9])
def test_real_rubric_size_is_bounded(count):
    row = clinical_case()
    row["rubrics"] = row["rubrics"][:count]
    while len(row["rubrics"]) < count:
        row["rubrics"].append({"id": f"extra-{len(row['rubrics'])}", "criterion": "Extra check", "points": 1})
    with pytest.raises(ValueError, match="4-8"):
        ClinicalCase.model_validate(row)


def test_real_rubric_requires_three_distinct_negative_criteria():
    row = clinical_case("uncertainty")
    row["rubrics"][3]["points"] = 1
    with pytest.raises(ValueError, match="at least three negative"):
        ClinicalCase.model_validate(row)


def test_base_negative_tags_cannot_collapse_onto_one_criterion():
    row = clinical_case("uncertainty")
    for criterion in row["rubrics"]:
        criterion["tags"] = []
    row["rubrics"][2]["tags"] = ["tempting_mistake", "unsupported_claim", "padding"]
    with pytest.raises(ValueError, match="three distinct negative criteria"):
        ClinicalCase.model_validate(row)


@pytest.mark.parametrize(
    "scenario,tag",
    [
        ("uncertainty", "tempting_mistake"),
        ("uncertainty", "unsupported_claim"),
        ("uncertainty", "padding"),
        ("urgent_escalation", "delayed_escalation"),
        ("context_seeking", "irrelevant_question"),
        ("documentation", "omitted_fact"),
        ("documentation", "invented_fact"),
    ],
)
def test_required_negative_behavior_tags_cannot_be_missing_or_only_positive(scenario, tag):
    row = clinical_case(scenario)
    for criterion in row["rubrics"]:
        criterion["tags"] = [value for value in criterion.get("tags", []) if value != tag]
    row["rubrics"][0]["tags"].append(tag)
    with pytest.raises(ValueError, match="missing required behavior tags"):
        ClinicalCase.model_validate(row)


@pytest.mark.parametrize("answer", [None, "", "  ", "The supplied source does not record the missing detail."])
def test_wrong_answer_must_be_nonempty_and_different(answer):
    with pytest.raises(ValueError, match="plausible_wrong_answer"):
        ClinicalCase.model_validate(clinical_case() | {"plausible_wrong_answer": answer})


@pytest.mark.parametrize("ids", [[], ["unknown"], ["faithful"], ["invented", "invented"]])
def test_contrast_failure_ids_must_reference_distinct_negative_criteria(ids):
    with pytest.raises(ValueError, match="contrast_failure_criterion_ids"):
        ClinicalCase.model_validate(clinical_case() | {"contrast_failure_criterion_ids": ids})


@pytest.mark.parametrize("turns", [1, 3, 5])
@pytest.mark.parametrize("system", [False, True])
def test_real_single_or_multiturn_histories_are_accepted(turns, system):
    row = clinical_case()
    row["messages"] = ([{"role": "system", "content": "Use supplied evidence."}] if system else []) + [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"Turn {index}"} for index in range(turns)
    ]
    assert ClinicalCase.model_validate(row).messages == row["messages"]


@pytest.mark.parametrize(
    "roles",
    [
        ["assistant", "user"],
        ["user", "user", "user"],
        ["user", "system", "user"],
        ["system", "system", "user"],
        ["user", "assistant", "user", "assistant", "user", "assistant", "user"],
    ],
)
def test_invalid_real_history_structure_fails(roles):
    with pytest.raises(ValueError, match="Clinical history"):
        ClinicalCase.model_validate(
            clinical_case() | {"messages": [{"role": role, "content": "Text"} for role in roles]}
        )


@pytest.mark.parametrize("mode, floor", [("rar_holistic", 0), ("additive", -1.5), ("protocol", 0)])
async def test_empty_policy_output_is_a_task_failure_without_judging(tmp_path, mode, floor):
    server = make_server(tmp_path, reward_mode=mode)
    with patch("resources_servers.clinical_consultation.app.call_judge", AsyncMock()) as judge:
        result = await server.verify(request(" "))
    assert result.reward == floor and result.evaluation_completed and not result.mask_sample
    assert result.judge_calls == 0
    judge.assert_not_called()


@pytest.mark.parametrize(
    "verdict", ["not JSON", '{"criteria_met": "true"}', '{"criteria_met": 1}', '{"score": 10}', "{}"]
)
async def test_invalid_judge_output_masks_after_bounded_attempts(tmp_path, verdict):
    server = make_server(tmp_path, reward_mode="additive", judge_max_attempts=2)
    with patch("resources_servers.clinical_consultation.app.call_judge", AsyncMock(return_value=completion(verdict))):
        result = await server.verify(request())
    assert result.mask_sample and not result.evaluation_completed
    assert result.judge_calls == 2
    assert result.model_dump()["_ng_failure_class"] == "judge_failed"
    assert result.model_dump()["instance_config"]["mask_sample"]


@pytest.mark.parametrize("error", [JudgeError("Unavailable"), TimeoutError("Late")])
async def test_transport_error_is_not_a_clinical_failure(tmp_path, error):
    server = make_server(tmp_path, judge_max_attempts=2)
    with patch("resources_servers.clinical_consultation.app.call_judge", AsyncMock(side_effect=error)):
        result = await server.verify(request())
    assert result.mask_sample and not result.evaluation_completed
    assert result.judge_calls == 2


async def test_additive_preserves_negative_raw_score(tmp_path):
    server = make_server(tmp_path, reward_mode="additive")
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(
            side_effect=[
                completion('{"criteria_met": false}'),
                completion('{"criteria_met": true}'),
            ]
        ),
    ):
        result = await server.verify(request())
    assert result.raw_score == -1.5 and result.reward == -1.5
    assert result.judge_calls == 2


async def test_additive_preserves_penalty_severity_ordering(tmp_path):
    bank_case = case() | {
        "rubrics": case()["rubrics"] + [{"id": "minor", "criterion": "Delays unnecessarily", "points": -1}]
    }
    server = make_server(tmp_path, cases=[bank_case], reward_mode="additive")
    rewards = []
    for flags in [(False, False, True), (False, True, False)]:
        with patch(
            "resources_servers.clinical_consultation.app.call_judge",
            AsyncMock(side_effect=[completion(json.dumps({"criteria_met": flag})) for flag in flags]),
        ):
            result = await server.verify(request())
        assert result.raw_score == result.reward
        rewards.append(result.reward)
    empty = await server.verify(request(""))
    assert empty.reward == -2 < rewards[1] == -1.5 < rewards[0] == -0.5
    assert all(reward < 0 for reward in rewards)


async def test_protocol_judges_conjunction_with_failure_clauses(tmp_path):
    server = make_server(tmp_path, reward_mode="protocol")
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(return_value=completion('{"criteria_met": false, "rationale": "Unsafe behavior."}')),
    ) as judge:
        result = await server.verify(request())
    prompt = judge.call_args.kwargs["json"].messages[0]["content"]
    assert "ALL" in prompt and "NO disqualifying" in prompt
    assert "Stops the simulation" in prompt and "Continues the simulation" in prompt
    assert result.reward == 0 and result.judge_calls == 1


def test_bank_hash_and_duplicates_are_checked(tmp_path):
    bank = tmp_path / "cases.jsonl"
    bank.write_text(json.dumps(case()) + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_case_bank(str(bank), "0" * 64)
    bank.write_text((json.dumps(case()) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_case_bank(str(bank), hashlib.sha256(bank.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    "protocols",
    [
        [{"id": "same", "criteria": ["stop"]}, {"id": "same", "criteria": ["stop"]}],
        [{"id": "bad", "criteria": ["unknown"]}],
        [{"id": "bad", "criteria": ["stop", "stop"]}],
        [{"id": "bad", "criteria": ["stop"], "failure_criteria": ["stop"]}],
        [{"id": "bad", "criteria": ["unsafe"]}],
    ],
)
def test_invalid_protocol_bank_fails_before_judging(tmp_path, protocols):
    bank = tmp_path / "cases.jsonl"
    bank.write_text(json.dumps(case() | {"protocols": protocols}) + "\n")
    with pytest.raises(ValueError, match="Protocol"):
        load_case_bank(str(bank), hashlib.sha256(bank.read_bytes()).hexdigest())


async def test_endpoint_concurrency_is_bounded_and_accounting_is_per_response(tmp_path):
    server = make_server(tmp_path, reward_mode="additive", judge_endpoint_max_concurrency=2)
    active = peak = 0

    async def judge(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        await asyncio.sleep(0.001)
        active -= 1
        criterion = kwargs["json"].messages[0]["content"]
        return completion(json.dumps({"criteria_met": not criterion.endswith("Continues the simulation")}))

    with patch("resources_servers.clinical_consultation.app.call_judge", side_effect=judge):
        results = await asyncio.gather(*(server.verify(request()) for _ in range(6)))
    assert peak == 2
    assert all(result.reward == 1 and result.judge_calls == 2 for result in results)


async def test_unknown_case_is_masked_without_endpoint_calls(tmp_path):
    server = make_server(tmp_path)
    result = await server.verify(request(case_id="unknown"))
    assert result.mask_sample and result.model_dump()["failure_kind"] == "clinical_consultation:case_bank_error"
    assert result.judge_calls == 0


async def test_simple_agent_keeps_private_case_off_policy_requests_and_retains_tokens(tmp_path):
    from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentRunRequest

    bank_case = clinical_case()
    verifier = make_server(tmp_path, cases=[bank_case])
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="127.0.0.1",
            port=11001,
            entrypoint="app.py",
            name="clinical_consultation_simple_agent",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            resources_server={"type": "resources_servers", "name": "clinical_consultation"},
        ),
        server_client=client,
    )
    policy_response = request().response.model_dump()
    policy_response["output"][0] |= {
        "prompt_token_ids": [10, 11],
        "generation_token_ids": [12, 13],
        "generation_log_probs": [-0.1, -0.2],
    }
    visible_requests = []

    def http_response(payload):
        response = MagicMock(status=200, cookies={}, ok=True)
        response.read = AsyncMock(return_value=json.dumps(payload))
        return response

    async def post(*, url_path, **kwargs):
        if url_path == "/seed_session":
            return http_response({})
        if url_path.endswith("/v1/responses"):
            payload = kwargs["json"]
            visible_requests.append(payload.model_dump() if hasattr(payload, "model_dump") else payload)
            return http_response(policy_response)
        assert url_path == "/verify"
        result = await verifier.verify(ClinicalVerifyRequest.model_validate(kwargs["json"]))
        return http_response(result.model_dump())

    client.post = AsyncMock(side_effect=post)
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(return_value=completion('{"score":10}')),
    ):
        result = await agent.run(
            MagicMock(cookies={}),
            SimpleAgentRunRequest.model_validate(
                {
                    "case_id": "fixture-1",
                    "responses_create_params": {"input": bank_case["messages"]},
                }
            ),
        )
    assert len(visible_requests) == 1
    assert not (
        {
            "case_id",
            "reference_answer",
            "rubrics",
            "patient",
            "provenance",
            "scenario",
            "benchmark_bucket",
            "user_type",
            "key_decision_criterion_id",
            "plausible_wrong_answer",
            "contrast_failure_criterion_ids",
        }
        & visible_requests[0].keys()
    )
    assert "Distinguishes recorded facts" not in json.dumps(visible_requests[0])
    assert result.reward == 1
    assert result.response.output[0].generation_token_ids == [12, 13]
    assert result.response.output[0].generation_log_probs == [-0.1, -0.2]
