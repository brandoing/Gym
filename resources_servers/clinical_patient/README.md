# Clinical patient evidence selection

This bounded training environment reveals evidence selected by the policy, then
grades a terminal urgency label and plan. It reuses `gymnasium_agent` and the
clinical consultation scorer. It tests **keyed evidence selection and completion**,
not free-text patient clarification, real patient behavior, or clinical validity.
No separately hosted patient model is required. Configurations remain unverified.

## Episode

1. `/reset` loads the case identified by the task's opaque `case_id`. The policy sees
   the initial observation and a bank-wide menu of evidence keys and urgency labels.
   Private values, references, rubric criteria and required-fact labels are not disclosed.
2. `ask_patient(question_key)` returns only that fact. Unknown keys return an
   unavailable-evidence observation. Repeated requests are counted, not rewarded.
3. `finish_consultation(urgency, plan)` ends the episode. Required evidence must have
   been requested and urgency must match the case contract. Urgent-escalation cases
   require completion on the first turn. These are synthetic task gates, not a medical verifier.
4. The terminal judge reads the visible interaction and final plan against
   `patient.reference_plan` and the case rubric. Missing evidence, incorrect urgency,
   delayed escalation or failure to finish within six assistant turns scores zero.
   Unavailable judging masks the episode instead of treating it as a clinical failure.

The server tracks state per session and the Gymnasium lifecycle closes completed,
truncated and failed episodes. In-flight sessions are restart-only; training recovery
uses the last complete update. Offline stateless re-verification is unsupported.
Every assistant response retains its original output items and token/log-probability
metadata; patient tool outputs are observations, not assistant training targets.

## Data and configuration

Use the same authenticated case bank as `clinical_consultation`, with `CASE_BANK_PATH`
and `CASE_BANK_SHA256`. Public rows contain `case_id`, `responses_create_params`
and an `agent_ref` named `clinical_patient_agent`. Provide the two function schemas
in `responses_create_params.tools`, with `parallel_tool_calls: false`:

- `ask_patient`: one string `question_key`.
- `finish_consultation`: string `urgency` and `plan`.

Both schemas disallow additional properties and require all listed arguments.
They contain no case-specific answers or private references. The policy must issue
exactly one supported function call per assistant turn.

Provide the remote judge through `JUDGE_BASE_URL`, `JUDGE_MODEL_NAME`,
`JUDGE_MODEL_REVISION` and private `JUDGE_API_KEY`. `CLINICAL_REWARD_MODE` accepts
`rar_holistic` or `likert`; the final judge runs once per eligible completed episode,
before bounded retries. Preserve judge usage/cost alongside policy training cost.

Compose `resources_servers/clinical_patient/configs/clinical_patient.yaml` with the
policy model configuration. NeMo RL uses
`responses_api_models/vllm_model/configs/vllm_model_for_training.yaml` to retain
training token information. The agent and environment both enforce six turns.
Qualify context length and total generation cost with a multi-turn canary separately
from the single-response runtime; do not infer capacity from a static task.

From the configured checkout environment:

```bash
python -m pytest resources_servers/clinical_patient/tests -q
```

The component tests exercise real transitions and scoring boundaries with a scripted
external judge. Live policy/judge rollouts and optimizer validation are separate gates.
Fictional engineering fixtures are not clinician-approved clinical training data.
