# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.clinical_consultation.tests.test_app import case, clinical_case, completion
from resources_servers.clinical_patient.app import ClinicalPatientConfig, ClinicalPatientServer


def patient_case(**overrides):
    value = case()
    value["reference_answer"] = "Ask for the signal color before deciding."
    value["messages"] = [{"role": "user", "content": "Fictional simulation: signal color is not provided. What next?"}]
    value["patient"] |= {
        "initial_observation": "Signal color is not provided.",
        "required_facts": ["color"],
        "expected_urgency": "routine",
    }
    return value | overrides


def make_server(tmp_path, *, cases=None, **overrides):
    bank = tmp_path / "cases.jsonl"
    bank.write_text("".join(json.dumps(row) + "\n" for row in (cases or [patient_case()])))
    config = ClinicalPatientConfig(
        host="127.0.0.1",
        port=11000,
        entrypoint="app.py",
        case_bank_path=str(bank),
        case_bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
        judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        judge_chat_completion_params={"messages": [], "temperature": 0, "max_tokens": 1024},
        question_keys=["color"],
        urgency_labels=["routine", "urgent"],
        **overrides,
    )
    return ClinicalPatientServer(config=config, server_client=MagicMock(spec=ServerClient))


def action(name, arguments, *, token=12):
    return NeMoGymResponse.model_validate(
        {
            "id": f"response-{token}",
            "created_at": 0,
            "model": "fixture",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": f"item-{token}",
                    "type": "function_call",
                    "name": name,
                    "call_id": f"call-{token}",
                    "arguments": json.dumps(arguments),
                    "status": "completed",
                    "prompt_token_ids": [10, 11],
                    "generation_token_ids": [token],
                    "generation_log_probs": [-0.1],
                }
            ],
        }
    )


async def test_conditional_disclosure_and_full_transcript_terminal_grading(tmp_path):
    server = make_server(tmp_path)
    observation, info = await server.reset({"case_id": "fixture-1"}, "one")
    assert "red" not in observation and "reference_plan" not in observation
    assert "color" in observation and info["case_id"] == "fixture-1"
    _, reward, terminated, truncated, info = await server.step(
        action("ask_patient", {"question_key": "color"}), {"case_id": "fixture-1"}, "one"
    )
    assert reward == 0 and not terminated and not truncated
    assert json.loads(info["tool_outputs"][0]["output"]) == {"question_key": "color", "answer": "red"}
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(return_value=completion('{"score":10}')),
    ) as judge:
        _, reward, terminated, truncated, info = await server.step(
            action("finish_consultation", {"urgency": "routine", "plan": "Stop the simulation."}),
            {"case_id": "fixture-1"},
            "one",
        )
    assert reward == 1 and terminated and not truncated
    assert info["evidence_complete"] and info["judge_calls"] == 1
    assert server.config.reward_mode == "rar_holistic"
    context = json.loads(judge.call_args.kwargs["json"].messages[1]["content"])
    assert "ask_patient" in context["conversation"] and "red" in context["conversation"]
    assert context["reference_answer"] == patient_case()["patient"]["reference_plan"]
    assert context["reference_answer"] != patient_case()["reference_answer"]
    await server.close_session("one")
    assert "one" not in server.session_state


async def test_cannot_finish_without_required_evidence_or_correct_urgency(tmp_path):
    server = make_server(tmp_path)
    await server.reset({"case_id": "fixture-1"}, "one")
    with patch("resources_servers.clinical_consultation.app.call_judge", AsyncMock()) as judge:
        _, reward, terminated, _, info = await server.step(
            action("finish_consultation", {"urgency": "routine", "plan": "Stop."}), {"case_id": "fixture-1"}, "one"
        )
    assert reward == 0 and terminated and not info["evidence_complete"]
    assert not info["mask_sample"] and info["evaluation_completed"]
    judge.assert_not_called()


@pytest.mark.parametrize("mode", ["rar_holistic", "additive", "protocol"])
async def test_known_information_can_complete_without_questioning(tmp_path, mode):
    row = patient_case()
    row["patient"]["required_facts"] = []
    server = make_server(tmp_path, cases=[row], reward_mode=mode)
    await server.reset({"case_id": "fixture-1"}, "one")
    with patch(
        "resources_servers.clinical_consultation.app.call_judge",
        AsyncMock(
            side_effect=[completion('{"score":10}')]
            if mode == "rar_holistic"
            else [completion('{"criteria_met":true}'), completion('{"criteria_met":false}')]
        ),
    ):
        _, reward, terminated, _, info = await server.step(
            action("finish_consultation", {"urgency": "routine", "plan": "Stop."}), {"case_id": "fixture-1"}, "one"
        )
    assert reward == 1 and terminated and info["asked_keys"] == []
    assert info["judge_calls"] == (2 if mode == "additive" else 1)


def test_patient_rejects_plain_likert(tmp_path):
    with pytest.raises(ValueError, match="reward_mode"):
        make_server(tmp_path, reward_mode="likert")


@pytest.mark.parametrize("failure", ["missing_evidence", "turn_limit", "invalid_at_limit"])
async def test_additive_task_failures_cannot_outscore_negative_answers(tmp_path, failure):
    server = make_server(tmp_path, reward_mode="additive")
    await server.reset({"case_id": "fixture-1"}, "one")
    if failure == "missing_evidence":
        response = action("finish_consultation", {"urgency": "routine", "plan": "Stop."})
    else:
        server.session_state["one"]["turns"] = 5
        response = action("ask_patient", {"question_key": "color"})
        if failure == "invalid_at_limit":
            response.output[0].arguments = "not JSON"
    _, reward, terminated, truncated, info = await server.step(response, {"case_id": "fixture-1"}, "one")
    row = patient_case()
    lower_bound = sum(c["points"] for c in row["rubrics"] if c["points"] < 0) / sum(
        c["points"] for c in row["rubrics"] if c["points"] > 0
    )
    assert reward == lower_bound < 0
    assert terminated or truncated
    assert info["evaluation_completed"] and not info["mask_sample"]


async def test_urgent_case_does_not_reward_delayed_escalation(tmp_path):
    row = patient_case(scenario="urgent_escalation")
    row["patient"] |= {"required_facts": [], "expected_urgency": "urgent"}
    server = make_server(tmp_path, cases=[row])
    await server.reset({"case_id": "fixture-1"}, "one")
    await server.step(action("ask_patient", {"question_key": "color"}), {"case_id": "fixture-1"}, "one")
    _, reward, terminated, _, info = await server.step(
        action("finish_consultation", {"urgency": "urgent", "plan": "Stop."}), {"case_id": "fixture-1"}, "one"
    )
    assert reward == 0 and terminated and not info["escalation_timely"]


async def test_repeated_questions_truncate_at_six_turns(tmp_path):
    server = make_server(tmp_path)
    await server.reset({"case_id": "fixture-1"}, "one")
    for _ in range(6):
        _, reward, terminated, truncated, info = await server.step(
            action("ask_patient", {"question_key": "color"}), {"case_id": "fixture-1"}, "one"
        )
    assert reward == 0 and not terminated and truncated
    assert info["turns"] == 6 and info["repeated_questions"] == 5
    _, _, terminated, _, info = await server.step(
        action("finish_consultation", {"urgency": "routine", "plan": "Stop."}), {"case_id": "fixture-1"}, "one"
    )
    assert terminated and info["mask_sample"]


async def test_sessions_are_isolated_and_unknown_questions_reveal_nothing(tmp_path):
    row2 = patient_case(case_id="fixture-2", family_id="family-2")
    row2["patient"]["facts"] = {"color": "blue"}
    server = make_server(tmp_path, cases=[patient_case(), row2])
    await server.reset({"case_id": "fixture-1"}, "one")
    await server.reset({"case_id": "fixture-2"}, "two")
    _, _, _, _, info1 = await server.step(
        action("ask_patient", {"question_key": "color"}), {"case_id": "fixture-1"}, "one"
    )
    _, _, _, _, info2 = await server.step(
        action("ask_patient", {"question_key": "color"}), {"case_id": "fixture-2"}, "two"
    )
    assert "red" in info1["tool_outputs"][0]["output"] and "blue" in info2["tool_outputs"][0]["output"]
    _, _, _, _, info = await server.step(
        action("ask_patient", {"question_key": "reference_answer"}), {"case_id": "fixture-1"}, "one"
    )
    assert "unavailable" in info["tool_outputs"][0]["output"]
    assert server.session_state["two"]["turns"] == 1


async def test_protected_cases_cannot_change_training_observation(tmp_path):
    train = patient_case()
    audit = patient_case(case_id="audit-only", family_id="audit-family", split="audit")
    audit["patient"]["facts"]["audit_private_key"] = "private audit value"
    audit["patient"]["expected_urgency"] = "audit_private_urgency"
    without_audit = make_server(tmp_path, cases=[train])
    with_audit = make_server(tmp_path, cases=[train, audit])
    first, first_info = await without_audit.reset({"case_id": "fixture-1"}, "train")
    second, second_info = await with_audit.reset({"case_id": "fixture-1"}, "train")
    assert first == second and first_info == second_info
    assert "audit_private_key" not in second and "audit_private_urgency" not in second
    with pytest.raises(ValueError, match="public patient ontology") as error:
        await with_audit.reset({"case_id": "audit-only"}, "audit")
    assert "audit_private_key" not in str(error.value) and "audit_private_urgency" not in str(error.value)


@pytest.mark.parametrize("scenario,bucket", [("documentation", "documentation"), ("evidence_research", "research")])
async def test_response_only_cases_cannot_start_patient_episode(tmp_path, scenario, bucket):
    response_only = clinical_case(scenario) | {"case_id": "response-only"}
    response_only.update(benchmark_bucket=bucket, user_type="clinician", patient=None)
    response_only["provenance"]["source_kind"] = "approved_source_excerpt"
    server = make_server(tmp_path, cases=[patient_case(), response_only])
    with pytest.raises(ValueError, match="Response-only case"):
        await server.reset({"case_id": "response-only"}, "wrong-route")
    assert "wrong-route" not in server.session_state
    observation, _ = await server.reset({"case_id": "fixture-1"}, "patient-route")
    assert "color" in observation


async def test_undeclared_bank_fact_is_not_a_policy_tool(tmp_path):
    row = patient_case()
    row["patient"]["facts"]["private_annotation"] = "This must stay private."
    server = make_server(tmp_path, cases=[row])
    observation, _ = await server.reset({"case_id": "fixture-1"}, "one")
    assert "private_annotation" not in observation
    _, _, _, _, info = await server.step(
        action("ask_patient", {"question_key": "private_annotation"}), {"case_id": "fixture-1"}, "one"
    )
    assert json.loads(info["tool_outputs"][0]["output"])["answer"] == "That evidence is unavailable."


async def test_judge_error_remains_masked_in_terminal_info(tmp_path):
    server = make_server(tmp_path, judge_max_attempts=1)
    await server.reset({"case_id": "fixture-1"}, "one")
    await server.step(action("ask_patient", {"question_key": "color"}), {"case_id": "fixture-1"}, "one")
    with patch(
        "resources_servers.clinical_consultation.app.call_judge", AsyncMock(side_effect=JudgeError("Unavailable"))
    ):
        _, reward, terminated, _, info = await server.step(
            action("finish_consultation", {"urgency": "routine", "plan": "Stop."}), {"case_id": "fixture-1"}, "one"
        )
    assert reward == 0 and terminated and info["mask_sample"] and not info["evaluation_completed"]
    assert info["judge_calls"] == 1 and info["instance_config"]["mask_sample"]


async def test_malformed_action_and_case_change_are_distinct_failures(tmp_path):
    server = make_server(tmp_path)
    await server.reset({"case_id": "fixture-1"}, "one")
    bad = action("ask_patient", {"question_key": "color"})
    bad.output[0].arguments = "not JSON"
    observation, reward, terminated, truncated, info = await server.step(bad, {"case_id": "fixture-1"}, "one")
    assert "JSON" in observation and reward == 0 and not terminated and not truncated
    assert not info["mask_sample"] and info["turns"] == 1
    assert info["tool_outputs"] == [{"call_id": "call-12", "output": json.dumps({"error": observation})}]
    _, _, terminated, _, info = await server.step(
        action("ask_patient", {"question_key": "color"}), {"case_id": "other"}, "one"
    )
    assert terminated and info["mask_sample"] and not info["evaluation_completed"]


async def test_invalid_multiple_calls_all_receive_error_outputs(tmp_path):
    server = make_server(tmp_path)
    await server.reset({"case_id": "fixture-1"}, "one")
    response = action("ask_patient", {"question_key": "color"}, token=12)
    response.output.extend(action("unknown_tool", {}, token=13).output)
    observation, _, terminated, truncated, info = await server.step(response, {"case_id": "fixture-1"}, "one")
    assert not terminated and not truncated
    assert [row["call_id"] for row in info["tool_outputs"]] == ["call-12", "call-13"]
    assert all(json.loads(row["output"])["error"] == observation for row in info["tool_outputs"])
    assert server.session_state["one"]["asked_keys"] == []


def test_http_lifecycle_closes_completed_episode(tmp_path):
    from fastapi.testclient import TestClient

    server = make_server(tmp_path)
    server.server_client.global_config_dict = {}
    with TestClient(server.setup_webserver()) as client:
        body = {"case_id": "fixture-1", "responses_create_params": {"input": patient_case()["messages"]}}
        reset = client.post("/reset", json=body)
        assert reset.status_code == 200
        assert len(server.session_state) == 1
        finish = client.post(
            "/step",
            json=body
            | {"response": action("finish_consultation", {"urgency": "routine", "plan": "Stop."}).model_dump()},
        )
        assert finish.status_code == 200 and finish.json()["terminated"]
        assert server.session_state == {}


async def test_agent_preserves_every_assistant_token_and_promotes_terminal_mask(tmp_path):
    from resources_servers.gymnasium.base import EnvResetResponse
    from responses_api_agents.gymnasium_agent.app import GymnasiumAgent, GymnasiumAgentConfig, GymnasiumAgentRunRequest

    verifier = make_server(tmp_path, judge_max_attempts=1)
    obs, info = await verifier.reset({"case_id": "fixture-1"}, "episode")
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    agent = GymnasiumAgent(
        config=GymnasiumAgentConfig(
            host="127.0.0.1",
            port=11001,
            entrypoint="app.py",
            name="clinical_patient_agent",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            resources_server={"type": "resources_servers", "name": "clinical_patient"},
            max_steps=6,
            text_only_history=False,
        ),
        server_client=client,
    )
    policy_calls = [
        action("ask_patient", {"question_key": "color"}, token=12),
        action("finish_consultation", {"urgency": "routine", "plan": "Stop."}, token=13),
    ]
    agent._request_model_response = AsyncMock(side_effect=[(response, {}) for response in policy_calls])

    async def post(*, url_path, **kwargs):
        assert url_path == "/step"
        _, reward, terminated, truncated, result_info = await verifier.step(
            NeMoGymResponse.model_validate(kwargs["json"]["response"]), kwargs["json"], "episode"
        )
        response = MagicMock(status=200, cookies={}, ok=True)
        response.read = AsyncMock(
            return_value=json.dumps(
                {"reward": reward, "terminated": terminated, "truncated": truncated, "info": result_info}
            )
        )
        return response

    client.post = AsyncMock(side_effect=post)
    body = GymnasiumAgentRunRequest.model_validate(
        {"case_id": "fixture-1", "responses_create_params": {"input": patient_case()["messages"]}}
    )
    with patch(
        "resources_servers.clinical_consultation.app.call_judge", AsyncMock(side_effect=JudgeError("Unavailable"))
    ):
        result = await agent._run_open_episode(body, "/v1/responses", EnvResetResponse(observation=obs, info=info), {})
    assert result.mask_sample and not result.evaluation_completed
    assert result.model_dump()["_ng_failure_class"] == "judge_failed"
    assistant_outputs = [item for item in result.response.output if item.type == "function_call"]
    assert [item.generation_token_ids for item in assistant_outputs] == [[12], [13]]
    assert all(item.generation_log_probs == [-0.1] for item in assistant_outputs)
    second_input = agent._request_model_response.call_args_list[1].args[0].input
    assert any(item.type == "function_call_output" for item in second_input)
