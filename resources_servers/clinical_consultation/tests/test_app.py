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
    VERIFIER_FIXTURE,
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
        "provenance": {"source_id": "fictional", "approved_for_training": False},
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


@pytest.mark.parametrize("removed_mode", ["likert", "rar_holistic"])
def test_overall_rating_modes_are_rejected(tmp_path, removed_mode):
    with pytest.raises(ValueError, match="reward_mode"):
        make_server(tmp_path, reward_mode=removed_mode)


async def test_each_prompt_uses_its_own_rubric(tmp_path):
    first = case()
    second = case() | {
        "case_id": "fixture-2",
        "family_id": "family-2",
        "rubrics": [{"id": "difference", "criterion": "Explains the absolute event difference", "points": 2}],
        "protocols": [],
    }
    server = make_server(tmp_path, cases=[first, second])
    assert server.config.reward_mode == "additive"
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(return_value=completion('{"criteria_met":true}')),
    ) as judge:
        await server.verify(request(case_id="fixture-2"))
    assert judge.call_count == 1
    prompt = judge.call_args.kwargs["json"].messages[0]["content"]
    assert "Explains the absolute event difference" in prompt and "Stops the simulation" not in prompt


async def test_request_cannot_override_private_bank(tmp_path):
    server = make_server(tmp_path)
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(return_value=completion('{"criteria_met": true}')),
    ) as judge:
        result = await server.verify(request(reference_answer="LEAK", rubrics=[{"criterion": "LEAK"}]))
    assert "LEAK" not in json.dumps(judge.call_args.kwargs["json"].model_dump())
    assert "reference_answer" not in result.model_dump()
    assert "rubrics" not in result.model_dump()


async def test_empty_policy_output_is_a_task_failure_without_judging(tmp_path):
    server = make_server(tmp_path)
    with patch("resources_servers.clinical_consultation.app.call_judge", AsyncMock()) as judge:
        result = await server.verify(request(" "))
    assert result.reward == 0 and result.evaluation_completed and not result.mask_sample
    assert result.judge_calls == 0
    judge.assert_not_called()


@pytest.mark.parametrize(
    "verdict", ["not JSON", '{"criteria_met": "true"}', '{"criteria_met": 1}', '{"score": 10}', "{}"]
)
async def test_invalid_judge_output_masks_after_bounded_attempts(tmp_path, verdict):
    server = make_server(tmp_path, judge_max_attempts=2)
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
    assert result.raw_score == -1.5 and result.reward == 0
    assert result.judge_calls == 2


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
    server = make_server(tmp_path, judge_endpoint_max_concurrency=2)
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

    verifier = make_server(tmp_path)
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
        AsyncMock(side_effect=[completion('{"criteria_met":true}'), completion('{"criteria_met":false}')]),
    ):
        result = await agent.run(
            MagicMock(cookies={}),
            SimpleAgentRunRequest.model_validate(
                {
                    "case_id": "fixture-1",
                    "responses_create_params": {"input": case()["messages"]},
                }
            ),
        )
    assert len(visible_requests) == 1
    assert not ({"case_id", "reference_answer", "rubrics", "patient", "provenance"} & visible_requests[0].keys())
    assert result.reward == 1
    assert result.response.output[0].generation_token_ids == [12, 13]
    assert result.response.output[0].generation_log_probs == [-0.1, -0.2]
