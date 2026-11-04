# Clinical consultation training rewards

This environment scores independently prepared clinical tasks using a private case bank.
It does not consume HealthBench data or change benchmark grading. New configurations
remain `verified: false`; fixture tests are not evidence of clinical quality or training gains.

## Inputs and visibility

Set `CASE_BANK_PATH` to a UTF-8 JSONL bank and `CASE_BANK_SHA256` to its SHA-256.
Each bank row contains `case_id`, `family_id`, `split`, `scenario`, `messages`,
`reference_answer`, `rubrics` (`id`, `criterion`, `points`), `protocols`
(`id`, `criteria`, `failure_criteria`), `patient`, and `provenance`.
The patient object contains `initial_observation`, `facts`, `required_facts`,
`expected_urgency`, and `reference_plan`. The server authenticates and validates
the bank at startup. Policy rows contain only the opaque `case_id`, public
`responses_create_params.input`, and the `clinical_consultation_simple_agent` reference.
Never put references or rubric criteria in the policy input.

## Reward modes

Select `CLINICAL_REWARD_MODE`:

- `likert`: one reference-conditioned overall quality judgement, without the case rubric.
- `rar_holistic`: the same judgement, with the case rubric. Both modes use a 1–10
  raw score mapped to `(score - 1) / 9`; they differ only in case-specific rubric access.
- `additive`: one boolean judgement per criterion, signed points divided by positive
  possible points. Raw scores remain available; only training rewards are clipped to [0,1].
- `protocol`: one boolean judgement per protocol dimension. A dimension requires
  all its positive requirements and no failure clause; reward is the proportion of
  dimensions passed. This is a protocol-inspired aggregation, not a reproduction claim.

Judge criteria and references are server-side. Candidate text is framed as untrusted
data. Empty policy answers score zero; unavailable or unusable judgements are masked
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
