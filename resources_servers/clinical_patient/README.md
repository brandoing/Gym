# Clinical patient evidence selection

This bounded training environment reveals evidence selected by the policy, then
grades a terminal urgency label and plan. It reuses `gymnasium_agent` and the
clinical consultation scorer. It tests **keyed evidence selection and completion**,
not free-text patient clarification, real patient behavior, or clinical validity.
No separately hosted patient model is required. Configurations remain unverified.

## Episode

1. `/reset` loads the case identified by the task's opaque `case_id`. The policy sees
   the initial observation and an explicitly configured menu of evidence keys and urgency labels.
   Private values, references, rubric criteria and required-fact labels are not disclosed.
2. `ask_patient(question_key)` returns only that fact. Unknown keys return an
   unavailable-evidence observation. Repeated requests are counted, not rewarded.
3. `finish_consultation(urgency, plan)` ends the episode. Required evidence must have
   been requested and urgency must match the case contract. Urgent-escalation cases
   require completion on the first turn. These are synthetic task gates, not a medical verifier.
4. The terminal judge checks the prompt's rubric criteria against the visible
   interaction and final plan, grounded in `patient.reference_plan`. Missing evidence, incorrect urgency,
   delayed escalation or failure to finish within six assistant turns receives the
   mode's minimum reward (zero for holistic/protocol; the case's negative-weight
   lower bound for additive). Abandoning a task cannot beat a penalized response.
   Unavailable judging masks the episode instead of treating it as a clinical failure.

The server tracks state per session and the Gymnasium lifecycle closes completed,
truncated and failed episodes. In-flight sessions are restart-only; training recovery
uses the last complete update. Offline stateless re-verification is unsupported.
Every assistant response retains its original output items and token/log-probability
metadata; patient tool outputs are observations, not assistant training targets.

## Data and configuration

Set `question_keys` and `urgency_labels` in the environment config before training.
They are a public action ontology, independent of the private case bank. Development
or audit cases must not expand the training menu. Undeclared evidence keys are never
revealed, even when present in a case. Reset rejects a case whose required evidence
or expected urgency cannot be represented by the configured ontology, without
including its hidden keys or labels in the error message.

Use the same authenticated case-bank format as `clinical_consultation`, with `CASE_BANK_PATH`
and `CASE_BANK_SHA256`. Public rows contain `case_id`, `responses_create_params`
and an `agent_ref` named `clinical_patient_agent`. Provide the two function schemas
in `responses_create_params.tools`, with `parallel_tool_calls: false`:

The bank may also contain response-only documentation, research or consultation
cases without patient metadata. They belong in the fixed-history response export;
interactive export includes only cases with an explicit patient contract. Reset
rejects response-only rows before creating a session. Do not invent patient facts
or an urgency decision simply to make a documentation or research task interactive.

- `ask_patient`: one string `question_key`.
- `finish_consultation`: string `urgency` and `plan`.

Both schemas disallow additional properties and require all listed arguments.
They contain no case-specific answers or private references. The policy must issue
exactly one supported function call per assistant turn.
The included fictional example bank is `data/example-casebank.jsonl`; its labels
use the example config's public ontology. Compute and supply its SHA-256 before use.

Provide the remote judge through `JUDGE_BASE_URL`, `JUDGE_MODEL_NAME`,
`JUDGE_MODEL_REVISION` and private `JUDGE_API_KEY`. `CLINICAL_REWARD_MODE` accepts
`rar_holistic` (default), `additive` or `protocol`. Holistic scoring reads the whole
case-specific rubric and returns one 1–10 score, normalized to 0–1. It is not a
rubric-free quality judge. Additive checks each criterion and preserves the signed
aggregate without clipping; protocol checks conjunctive dimensions. The number
of judge calls depends on those checks and bounded retries. Preserve judge usage/cost
alongside policy training cost.

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
