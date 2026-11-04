# Clinical consultation training rewards

This environment scores independently prepared clinical tasks using a private case bank.
It does not consume HealthBench data or change benchmark grading. New configurations
remain `verified: false`; fixture tests are not evidence of clinical quality or training gains.

## Inputs and visibility

Set `CASE_BANK_PATH` to a UTF-8 JSONL bank and `CASE_BANK_SHA256` to its SHA-256.
Each real bank row contains `case_id`, `family_id`, `split`, `scenario`,
`benchmark_bucket`, `user_type`, `messages`, `reference_answer`, `rubrics`
(`id`, `criterion`, `points`, optional `tags`), `key_decision_criterion_id`,
`plausible_wrong_answer`, `contrast_failure_criterion_ids`, and `provenance`. `protocols`
(`id`, `criteria`, `failure_criteria`) and `patient` are optional. When present,
the patient object contains `initial_observation`, `facts`, `required_facts`,
`expected_urgency`, and `reference_plan`. Documentation and research response tasks
do not need to invent patient facts or an urgency decision. The server authenticates and validates
the bank at startup. Policy rows contain only the opaque `case_id`, public
`responses_create_params.input`, and the `clinical_consultation_simple_agent` reference.
Never put references, rubric criteria or private task labels in the policy input.

Real cases use one of these scenarios:

| Scenario | Intended behavior |
|---|---|
| `context_seeking` | Ask for missing information that changes the recommendation |
| `complete_information` | Answer directly when sufficient information is present |
| `urgent_escalation` | Put emergency action first |
| `safety_trap` | Stop, avoid or correct a harmful proposed action |
| `documentation` | Transform supplied encounter facts faithfully into the requested document |
| `evidence_research` | Critique or compare evidence and its limits |
| `adversarial` | Correct false premises or resist unsafe pressure |
| `global_health` | Account for the stated local setting and resource constraints |
| `audience_tailoring` | Match depth and vocabulary to the actual user |
| `uncertainty` | Express uncertainty warranted by missing or conflicting evidence |

`benchmark_bucket` is `consult`, `documentation` or `research`; `user_type` is
`clinician` or `patient`. A `documentation` scenario requires the documentation
bucket and `evidence_research` requires research. The other scenarios describe
behaviors that can occur across buckets. Every real case must supply both tags;
they are interpreted by the judge alongside that case's own rubric and reference,
not used as a generic score or shown to the policy.

The old `missing_information` / `evidence_interpretation` labels and the example
`simulation` label are accepted only when `provenance.source_kind` is
`fictional_engineering_fixture` and `approved_for_training` is false. This preserves
wiring fixtures without admitting untagged legacy cases as real clinical training
data. Existing real banks need explicit taxonomy tags; no silent migration occurs.
These categories organize independent training tasks; they do not establish
benchmark coverage, clinical validity or transfer to HealthBench/Professional.

Real cases must have 4–8 criteria, at least three of them negative. The declared
`key_decision_criterion_id` identifies a positive criterion with at least the
weight of every other positive criterion. Negative criteria collectively cover
`tempting_mistake`, `unsupported_claim` and `padding` tags on three distinct
negative criterion IDs. Urgent escalation also
requires `delayed_escalation`; context seeking requires `irrelevant_question`;
documentation requires both `omitted_fact` and `invented_fact`. A criterion may
cover multiple tags. The actual criterion text still needs source and clinical
review: merely supplying a tag does not establish the intended behavior.

The private `plausible_wrong_answer` differs from the reference, and
`contrast_failure_criterion_ids` lists the negative criteria it is intended to
trigger. These fields support a separate reference-versus-wrong-answer judge
qualification check. They are neither policy input nor supplied as a comparison
answer during normal policy grading, and schema validation does not establish
that the judge actually distinguishes them.

Real conversations contain either a single user message or 2–4 prior messages
before the final user message, alternating user/assistant, with an optional
initial system message. Dataset-level history proportions are generation/export
constraints. The verifier checks each history's structure but cannot establish
that a whole training split has the intended mix. Wiring fixtures retain their
existing small rubrics and do not acquire these clinical content requirements.

## Reward modes

Every prompt has its own private rubric. Select `CLINICAL_REWARD_MODE`:

- `rar_holistic` (default): the judge reads the conversation, reference answer and
  complete case-specific rubric, then gives one overall 1–10 score. `raw_score`
  retains that score; training reward is `(score - 1) / 9`. One call per response
  before retries. This is rubric-conditioned scoring, not a generic Likert control.
- `additive` (secondary): one boolean judgement per criterion, signed points divided
  by positive possible points. The same signed value is returned as `raw_score` and
  training reward, without clipping; more severe penalties remain distinguishable.
  Its lower bound is the sum of negative points divided by positive possible points.
- `protocol` (optional): one boolean judgement per protocol dimension. A dimension requires
  all its positive requirements and no failure clause; reward is the proportion of
  dimensions passed. This mode requires protocol definitions in every evaluated
  case; missing definitions are an infrastructure/configuration failure, not a zero
  score for the policy. This is a protocol-inspired aggregation, not a reproduction claim.

The primary arm follows the implicit rubric-scoring approach in
[Rubrics as Rewards, sections 2.2 and 4.4](https://arxiv.org/html/2507.17746v2).
This is a RaR-style adaptation using the private reference and signed criteria,
not an exact reproduction. The paper reports application-dependent tradeoffs
between implicit and explicit aggregation; additive remains a useful secondary arm.

Plain `likert` is unsupported. Compare the trained checkpoint with the unchanged
model and matched controls; this design alone does not attribute gains specifically
to rubric conditioning. The holistic prompt is independently implemented and has
not yet been qualified with a live judge or shown to reproduce published results.

For example, an escalation prompt can require timely escalation and penalize
unnecessary delay; an evidence-interpretation prompt can require a correct absolute
difference and penalize an unsupported individual-outcome guarantee. These criteria
come from the selected case rather than a fixed generic rubric.

Judge criteria and references are server-side. Candidate text is framed as untrusted
data. Empty policy answers receive the selected mode's minimum reward (including
negative penalties in additive mode), so silence cannot beat a penalized answer.
Unavailable or unusable judgements are masked
infrastructure failures. Each client judge invocation records its check, attempt,
ID/model, token usage when returned, latency, and status. `judge_calls` counts these
invocations, including bounded retries; lower-level HTTP retries require endpoint telemetry.
Concurrency is per server process, not a deployment-wide quota or spend limit.

## Run

Provide `JUDGE_BASE_URL`, `JUDGE_MODEL_NAME`, and `JUDGE_API_KEY` privately. Set
`JUDGE_MODEL_REVISION` to the deployment's resolved revision; it is retained with
each judgement. A missing revision is recorded as unknown, not inferred from an alias. The judge
must be distinct from the training policy server. Compose this environment with the
model-server configuration appropriate to your policy:

```bash
gym env start \
  '+config_paths=[resources_servers/clinical_consultation/configs/clinical_consultation.yaml,responses_api_models/openai_model/configs/openai_model.yaml]' \
  '+policy_base_url=${oc.env:POLICY_BASE_URL}' \
  '+policy_model_name=${oc.env:POLICY_MODEL_NAME}' \
  '+policy_api_key=${oc.env:POLICY_API_KEY}'
```

For NeMo RL, compose `responses_api_models/vllm_model/configs/vllm_model_for_training.yaml`
instead. That model configuration preserves training token information. Pin all
model, judge, bank and runtime identities before comparison. Keep training and
evaluation source/case families disjoint and obtain the required data/model approvals.

Run component tests from the checkout's configured environment:

```bash
python -m pytest resources_servers/clinical_consultation/tests -q
```

Tests replace the external judge and exercise scoring/failure behavior. Representative
live policy-and-judge rollouts and optimizer validation remain separate qualification steps.
