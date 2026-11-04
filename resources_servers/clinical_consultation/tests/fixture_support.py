# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic verifier onboarding; substitutes only the external judge."""

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.clinical_consultation.app import ClinicalConsultationConfig, ClinicalConsultationServer


def fixture_server() -> ClinicalConsultationServer:
    bank = Path(__file__).parents[1] / "data" / "example-casebank.jsonl"
    config = ClinicalConsultationConfig(
        host="127.0.0.1",
        port=11000,
        entrypoint="app.py",
        case_bank_path=str(bank),
        case_bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
        judge_model_server={"type": "responses_api_models", "name": "judge_model"},
        judge_chat_completion_params={"messages": [], "temperature": 0, "max_tokens": 1024},
    )
    server = ClinicalConsultationServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def judge(guide, context, *, check_id, records, numeric=False):
        answer = json.loads(context)["candidate"]
        criteria_met = answer == ("Continue the simulation." if check_id == "unsafe" else "Stop the simulation.")
        records.append({"check_id": check_id, "attempt": 1, "status": "fixture"})
        if numeric:
            return {"score": 10 if criteria_met else 1, "rationale": "Deterministic fixture judge"}
        return {"criteria_met": criteria_met, "rationale": "Deterministic fixture judge"}

    server._judge = AsyncMock(side_effect=judge)
    return server
