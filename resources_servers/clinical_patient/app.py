# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded evidence selection over private, deterministic patient case facts."""

from __future__ import annotations

import json
from typing import ClassVar, Literal

from pydantic import Field, JsonValue

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.clinical_consultation.app import ClinicalConsultationConfig, ClinicalConsultationServer
from resources_servers.gymnasium.base import GymnasiumServer


class ClinicalPatientConfig(ClinicalConsultationConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    name: str = "clinical_patient"
    reward_mode: Literal["likert", "rar_holistic"] = "rar_holistic"
    max_turns: int = Field(default=6, ge=1, le=6)


class ClinicalPatientServer(GymnasiumServer, ClinicalConsultationServer):
    """Reveal only requested evidence and grade the completed consultation."""

    config: ClinicalPatientConfig

    def model_post_init(self, context: object) -> None:
        # Gymnasium's generated private-attribute hook precedes the scorer in the MRO.
        ClinicalConsultationServer.model_post_init(self, context)

    async def reset(self, metadata: dict, session_id: str | None = None) -> tuple[str | None, dict]:
        """Create one isolated patient episode without revealing hidden values."""
        if not session_id:
            raise ValueError("Clinical patient reset requires an isolated session ID")
        case = self.get_case(metadata.get("case_id", ""))
        if session_id in self.session_state:
            state = self.session_state[session_id]
            if state["case_id"] != case.case_id or state["turns"]:
                raise ValueError("Patient session is already active")
        else:
            self.session_state[session_id] = {
                "case_id": case.case_id,
                "turns": 0,
                "asked_keys": [],
                "repeated_questions": 0,
                "unavailable_questions": 0,
                "done": False,
                "history": list(case.messages),
            }
        # A shared menu does not identify which evidence exists or matters in this case.
        keys = sorted({key for entry in self._cases.values() for key in entry.patient.facts})
        urgency_labels = sorted({entry.patient.expected_urgency for entry in self._cases.values()})
        observation = (
            case.patient.initial_observation
            + "\nAvailable evidence keys: "
            + ", ".join(keys)
            + ". Use ask_patient to select evidence, or finish_consultation to give urgency and plan."
            + " Valid urgency labels: "
            + ", ".join(urgency_labels)
            + "."
        )
        return observation, {"case_id": case.case_id, "max_turns": self.config.max_turns}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: str | None = None
    ) -> tuple[str | None, float, bool, bool, dict]:
        """Execute one evidence request or terminal decision, enforcing turn limits."""
        if session_id not in self.session_state:
            return (
                None,
                0,
                True,
                False,
                self.failure_fields("clinical_patient:missing_session", "Patient session is not active"),
            )
        state = self.session_state[session_id]
        if state["done"]:
            return (
                None,
                0,
                True,
                False,
                self.failure_fields("clinical_patient:closed_session", "Patient session is complete"),
            )
        if metadata.get("case_id") != state["case_id"]:
            state["done"] = True
            return (
                None,
                0,
                True,
                False,
                self.failure_fields("clinical_patient:case_mismatch", "Episode case ID changed"),
            )
        case = self.get_case(state["case_id"])
        state["turns"] += 1
        calls = [item for item in action.output if item.type == "function_call"]
        visible_action = action.output_text + " " + " ".join(call.name + " " + call.arguments for call in calls)
        state["history"].append({"role": "assistant", "content": visible_action.strip()})
        if len(calls) != 1:
            return self._invalid_action(state, "Use exactly one supported function call per turn.")
        call = calls[0]
        try:
            arguments = json.loads(call.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Arguments must be a JSON object")
        except (ValueError, TypeError):
            return self._invalid_action(state, "Function arguments must be a JSON object.")
        if call.name == "ask_patient":
            if set(arguments) != {"question_key"} or not isinstance(arguments["question_key"], str):
                return self._invalid_action(state, "ask_patient requires one string question_key.")
            key = arguments["question_key"]
            if key in state["asked_keys"]:
                state["repeated_questions"] += 1
            if key in case.patient.facts:
                if key not in state["asked_keys"]:
                    state["asked_keys"].append(key)
                answer = case.patient.facts[key]
            else:
                state["unavailable_questions"] += 1
                answer = "That evidence is unavailable."
            observation = {"question_key": key, "answer": answer}
            state["history"].append({"role": "user", "content": json.dumps(observation)})
            if state["turns"] >= self.config.max_turns:
                state["done"] = True
                return (
                    None,
                    0,
                    False,
                    True,
                    self._episode_info(state) | {"failure_kind": "clinical_patient:turn_limit"},
                )
            return (
                None,
                0,
                False,
                False,
                self._episode_info(state)
                | {
                    "tool_outputs": [{"call_id": call.call_id, "output": json.dumps(observation)}],
                },
            )
        if call.name != "finish_consultation" or set(arguments) != {"urgency", "plan"}:
            return self._invalid_action(state, "Use ask_patient or finish_consultation(urgency, plan).")
        if not all(isinstance(arguments[key], str) and arguments[key].strip() for key in ("urgency", "plan")):
            return self._invalid_action(state, "Completion requires nonempty urgency and plan strings.")
        state["done"] = True
        evidence_complete = set(case.patient.required_facts) <= set(state["asked_keys"])
        urgency_correct = arguments["urgency"].strip().lower() == case.patient.expected_urgency.strip().lower()
        escalation_timely = case.scenario != "urgent_escalation" or state["turns"] == 1
        info = self._episode_info(state) | {
            "evidence_complete": evidence_complete,
            "urgency_correct": urgency_correct,
            "escalation_timely": escalation_timely,
            "judge_calls": 0,
            "judge_records": [],
        }
        # These are synthetic task-contract gates, not a claim to verify medical truth.
        if not (evidence_complete and urgency_correct and escalation_timely):
            return None, 0, True, False, info | {"failure_kind": "clinical_patient:incomplete_decision"}
        terminal_case = case.model_copy(update={"reference_answer": case.patient.reference_plan})
        records: list[dict[str, JsonValue]] = []
        try:
            raw_score, reward = await self.grade(
                terminal_case,
                arguments["plan"],
                records=records,
                conversation=json.dumps(state["history"], ensure_ascii=False),
            )
        except JudgeError as error:
            return (
                None,
                0,
                True,
                False,
                info
                | self.failure_fields("judge_failed", str(error))
                | {
                    "judge_calls": len(records),
                    "judge_records": records,
                },
            )
        return (
            None,
            reward,
            True,
            False,
            info
            | {
                "raw_score": raw_score,
                "judge_calls": len(records),
                "judge_records": records,
            },
        )

    @staticmethod
    def _episode_info(state: dict) -> dict:
        return {
            "case_id": state["case_id"],
            "turns": state["turns"],
            "asked_keys": list(state["asked_keys"]),
            "repeated_questions": state["repeated_questions"],
            "unavailable_questions": state["unavailable_questions"],
            "evaluation_completed": True,
            "mask_sample": False,
        }

    def _invalid_action(self, state: dict, observation: str) -> tuple[str | None, float, bool, bool, dict]:
        truncated = state["turns"] >= self.config.max_turns
        state["done"] = truncated
        return (
            observation,
            0,
            False,
            truncated,
            self._episode_info(state) | {"failure_kind": "clinical_patient:invalid_action"},
        )


if __name__ == "__main__":
    ClinicalPatientServer.run_webserver()
