package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/strictjson"
)

const (
	evaluationPlanMediaType = "application/vnd.overgo.evaluation-plan+json"
	evaluationPlanSchema    = "overgo/evaluation-plan/v1"
	caseProfileMediaType    = "application/vnd.overgo.evaluation-case-profile+json"
	caseProfileSchema       = "overgo/evaluation-case-profile/v1"
	scorerProfileMediaType  = "application/vnd.overgo.evaluation-scorer+json"
	scorerProfileSchema     = "overgo/evaluation-scorer/v1"
	executionMediaType      = "application/vnd.overgo.evaluation-execution+json"
	executionSchema         = "overgo/evaluation-execution/v1"
)

const (
	EvaluationPlanMediaType = evaluationPlanMediaType
	EvaluationPlanSchema    = evaluationPlanSchema
)

var (
	evaluationPlanContract = artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: evaluationPlanMediaType, Schema: evaluationPlanSchema,
	}
	caseProfileContract = artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: caseProfileMediaType, Schema: caseProfileSchema,
	}
	scorerProfileContract = artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: scorerProfileMediaType, Schema: scorerProfileSchema,
	}
	executionContract = artifact.DocumentContract{
		Kind: artifact.KindProfile, MediaType: executionMediaType, Schema: executionSchema,
	}
)

type lifecycle string

const (
	LifecycleIsolated lifecycle = "isolated"
	LifecycleResident lifecycle = "resident"
)

type ExecutionPolicy struct {
	Lifecycle lifecycle `json:"lifecycle"`
	// Prompting names how a scored prompt reaches the model: the raw
	// completion text (the recorded anchor, the empty default so
	// earlier plans keep their identity) or the model's own declared
	// chat template. Two protocols are two plans, so their records
	// never collapse into one "latest" score.
	Prompting prompting `json:"prompting,omitzero"`
	// Reasoning is every setting the reasoning protocol asks under.
	//   - zero under every other protocol => absent from its execution document
	Reasoning reasoningPolicy `json:"reasoning,omitzero"`
}

// valid accepts only the lifecycle and prompting contracts the runtime owns.
//   - Reasoning set <=> Prompting = PromptingReasoning
func (policy ExecutionPolicy) valid() bool {
	reasons := policy.Prompting == PromptingReasoning
	return (policy.Lifecycle == LifecycleIsolated || policy.Lifecycle == LifecycleResident) &&
		(policy.Prompting == PromptingRawCompletion || policy.Prompting == PromptingChatTemplate || policy.Prompting == PromptingHostedChat || reasons) &&
		reasons == (policy.Reasoning != reasoningPolicy{}) && (!reasons || policy.Reasoning.valid())
}

// prompting is the prompt-shaping protocol an execution policy binds.
type prompting string

const (
	// PromptingRawCompletion scores the suite's prompt text unchanged.
	PromptingRawCompletion prompting = ""
	// PromptingChatTemplate shapes each prompt through the model's
	// declared chat template before scoring.
	PromptingChatTemplate prompting = "chat-template"
	// PromptingHostedChat sends each prompt as the user message of a
	// hosted provider's chat completion: the provider owns the template,
	// so the answer opener rides the user message and no likelihood is
	// observable; suites score generatively.
	PromptingHostedChat prompting = "hosted-chat"
	// PromptingReasoning asks each choice question through the model's
	// declared chat template and lets the model write up to a long limit
	// before an answer parser reads its choice (exploratoryReasoning).
	PromptingReasoning prompting = "exploratory-reasoning"
)

// Label names the protocol for reports; the raw default reads as such.
func (prompting prompting) Label() string {
	if prompting == PromptingRawCompletion {
		return "raw-completion"
	}
	return string(prompting)
}

// PromptingNamed returns the protocol a local model is asked under by its label.
//   - accepted = raw-completion, chat-template, exploratory-reasoning
//   - hosted-chat is refused because a hosted model's location decides it.
func PromptingNamed(label string) (prompting, error) {
	for _, known := range []prompting{PromptingRawCompletion, PromptingChatTemplate, PromptingReasoning} {
		if known.Label() == label {
			return known, nil
		}
	}
	return "", fmt.Errorf("evaluation: protocol %q is not one a local model is asked under; accepted are %s, %s and %s",
		label, PromptingRawCompletion.Label(), PromptingChatTemplate.Label(), PromptingReasoning.Label())
}

type ExactAuthorities struct {
	ModelDefinition artifact.ID
	RuntimeRecipe   artifact.ID
	CodeCommit      string
	Environment     artifact.ID
	Execution       ExecutionPolicy
	// AnswerScope names the code a case's answer is kept under where the
	// build counted an identity for it (AnsweredBy); empty keeps an answer
	// under CodeCommit. It is no part of a plan's identity.
	AnswerScope string
}

type Plan struct {
	identity    artifact.ID
	body        planBody
	authorities []artifact.Content
	// answered is the authorities' AnswerScope: what the plan's answers are
	// kept under in place of its commit (observed).
	answered string
	// reasoning is the bound execution policy's Reasoning, which a choice
	// suite is answered and parsed by.
	reasoning reasoningPolicy
}

type planBody struct {
	Version         uint16                      `json:"version"`
	ModelDefinition artifact.ID                 `json:"model_definition"`
	RuntimeRecipe   artifact.ID                 `json:"runtime_recipe"`
	Dataset         artifact.ID                 `json:"dataset"`
	Split           artifact.ID                 `json:"split"`
	CaseProfile     artifact.ID                 `json:"case_profile"`
	Scorer          artifact.ID                 `json:"scorer"`
	Execution       artifact.ID                 `json:"execution"`
	CodeCommit      string                      `json:"code_commit"`
	Environment     artifact.ID                 `json:"environment"`
	Agent           *agentTrajectoryAuthorities `json:"agent,omitempty"`
}

// agentTrajectoryScore names one deterministic, code-owned trajectory scorer.
type agentTrajectoryScore string

const (
	// agentScoreTaskSuccess names the scorer for task acceptance success.
	agentScoreTaskSuccess agentTrajectoryScore = "task-success"
	// agentScoreEvidenceCompleteness names the scorer for committed-evidence completeness.
	agentScoreEvidenceCompleteness agentTrajectoryScore = "evidence-completeness"
	// agentScoreGateAccuracy names the scorer for gate-result accuracy.
	agentScoreGateAccuracy agentTrajectoryScore = "gate-accuracy"
	// agentScoreEffectPrediction names the scorer for predicted-versus-observed effects.
	agentScoreEffectPrediction agentTrajectoryScore = "effect-prediction"
	// agentScoreRegressions names the scorer for introduced regressions.
	agentScoreRegressions agentTrajectoryScore = "regressions"
	// agentScoreChurn names the scorer for code churn spent on the task.
	agentScoreChurn agentTrajectoryScore = "churn"
	// agentScoreRepeatedDirections names the scorer for repeated attempt directions.
	agentScoreRepeatedDirections agentTrajectoryScore = "repeated-directions"
	// agentScoreBudget names the scorer for consumed budget.
	agentScoreBudget agentTrajectoryScore = "budget"
	// agentScoreLatency names the scorer for wall-clock latency.
	agentScoreLatency agentTrajectoryScore = "latency"
	// agentScoreUnsupportedCompletion names the scorer for completion claims without evidence.
	agentScoreUnsupportedCompletion agentTrajectoryScore = "unsupported-completion"
)

// agentTrajectoryAuthorities extend the exact plan. Deterministic scorers are
// code-owned hard facts; advisory judges are versioned profiles only.
type agentTrajectoryAuthorities struct {
	Trajectories   []artifact.ID          `json:"trajectories"`
	Deterministic  []agentTrajectoryScore `json:"deterministic"`
	AdvisoryJudges []artifact.ID          `json:"advisory_judges,omitempty"`
}

var requiredAgentTrajectoryScores = []agentTrajectoryScore{
	agentScoreTaskSuccess, agentScoreEvidenceCompleteness, agentScoreGateAccuracy, agentScoreEffectPrediction,
	agentScoreRegressions, agentScoreChurn, agentScoreRepeatedDirections, agentScoreBudget, agentScoreLatency,
	agentScoreUnsupportedCompletion,
}

// BindAgentTrajectoryPlan binds exact trajectories to the existing evaluation
// plan identity without granting model judges promotion authority.
func BindAgentTrajectoryPlan(base Plan, trajectories, advisoryJudges []artifact.ID) (Plan, error) {
	if err := base.ValidateIdentity(); err != nil {
		return Plan{}, err
	}
	agent := &agentTrajectoryAuthorities{Trajectories: slices.Clone(trajectories), Deterministic: slices.Clone(requiredAgentTrajectoryScores), AdvisoryJudges: slices.Clone(advisoryJudges)}
	if err := canonicalizeAgentTrajectoryAuthorities(agent); err != nil {
		return Plan{}, err
	}
	body := base.body
	body.Agent = agent
	identity, err := artifact.JSONID(artifact.KindProfile, body)
	if err != nil {
		return Plan{}, err
	}
	return Plan{identity: identity, body: body, authorities: slices.Clone(base.authorities)}, nil
}

func BindExact(exact ExactPlan, authorities ExactAuthorities) (Plan, error) {
	if !exact.identity.Valid() || exact.dataset.Kind() != artifact.KindDataset ||
		exact.split.Kind() != artifact.KindDatasetShard ||
		authorities.ModelDefinition.Kind() != artifact.KindModelDefinition ||
		authorities.RuntimeRecipe.Kind() != artifact.KindRecipe ||
		authorities.Environment.Kind() != artifact.KindEvidence ||
		!validCommit(authorities.CodeCommit) {
		return Plan{}, errors.New("evaluation: invalid exact authorities")
	}
	scorer := struct {
		Version uint16 `json:"version"`
		Kind    string `json:"kind"`
	}{Version: artifact.InitialDocumentVersion, Kind: "exact-generation"}
	return bindPlan(exact.dataset, exact.split, exact.identity, exact.suite, scorer, authorities)
}

// bindKindPlan binds a compiled suite whose scorer document carries
// only the document version and its kind; every parameterless scorer
// shares this shape.
func bindKindPlan[C any](dataset, split, caseProfileID artifact.ID, caseProfile C, kind string, authorities ExactAuthorities) (Plan, error) {
	scorer := struct {
		Version uint16 `json:"version"`
		Kind    string `json:"kind"`
	}{Version: artifact.InitialDocumentVersion, Kind: kind}
	return bindPlan(dataset, split, caseProfileID, caseProfile, scorer, authorities)
}

// bindPlan binds a suite of any kind but a choice suite.
//   - PromptingReasoning refused because only a choice suite's answers are parsed by it
func bindPlan[C, S any](
	dataset, split, caseProfileID artifact.ID,
	caseProfile C,
	scorer S,
	authorities ExactAuthorities,
) (Plan, error) {
	if authorities.Execution.Prompting == PromptingReasoning {
		return Plan{}, fmt.Errorf("evaluation: the %s protocol asks choice suites alone, and this suite's cases are not choices", PromptingReasoning.Label())
	}
	return bindAsked(dataset, split, caseProfileID, caseProfile, scorer, authorities)
}

// bindAsked binds a suite under any protocol its kind's binder admits.
func bindAsked[C, S any](
	dataset, split, caseProfileID artifact.ID,
	caseProfile C,
	scorer S,
	authorities ExactAuthorities,
) (Plan, error) {
	caseContent, err := authorityContent(caseProfileContract, caseProfile)
	if err != nil || caseContent.Descriptor.ID != caseProfileID {
		return Plan{}, errors.Join(err, errors.New("evaluation: case profile identity differs"))
	}
	scorerContent, err := authorityContent(scorerProfileContract, scorer)
	if err != nil {
		return Plan{}, err
	}
	executionContent, err := authorityContent(executionContract, authorities.Execution)
	if err != nil {
		return Plan{}, err
	}
	if dataset.Kind() != artifact.KindDataset || split.Kind() != artifact.KindDatasetShard ||
		caseProfileID.Kind() != artifact.KindProfile ||
		authorities.ModelDefinition.Kind() != artifact.KindModelDefinition ||
		authorities.RuntimeRecipe.Kind() != artifact.KindRecipe ||
		authorities.Environment.Kind() != artifact.KindEvidence || !validCommit(authorities.CodeCommit) {
		return Plan{}, errors.New("evaluation: invalid plan authorities")
	}
	if !authorities.Execution.valid() {
		return Plan{}, errors.New("evaluation: invalid execution policy")
	}
	body := planBody{
		Version:         artifact.InitialDocumentVersion,
		ModelDefinition: authorities.ModelDefinition, RuntimeRecipe: authorities.RuntimeRecipe,
		Dataset: dataset, Split: split, CaseProfile: caseProfileID,
		Scorer: scorerContent.Descriptor.ID, Execution: executionContent.Descriptor.ID, CodeCommit: authorities.CodeCommit,
		Environment: authorities.Environment,
	}
	identity, err := artifact.JSONID(artifact.KindProfile, body)
	if err != nil {
		return Plan{}, err
	}
	return Plan{
		identity: identity, body: body,
		authorities: []artifact.Content{caseContent, scorerContent, executionContent},
		answered:    authorities.AnswerScope, reasoning: authorities.Execution.Reasoning,
	}, nil
}

func (p Plan) Identity() artifact.ID { return p.identity }

func (p Plan) Dataset() artifact.ID { return p.body.Dataset }

// Execution names the plan's bound execution-policy authority.
func (p Plan) Execution() artifact.ID { return p.body.Execution }

// comparisonIdentity binds acceptance and environment for comparisons across
// producer/model changes. It does not authorize reuse: an answer is given
// again only to what the plan's execution was asked, by the code that gave it
// (observed).
func (p Plan) comparisonIdentity() (artifact.ID, error) {
	return artifact.JSONID(artifact.KindProfile, struct {
		Version     uint16                      `json:"version"`
		Dataset     artifact.ID                 `json:"dataset"`
		Split       artifact.ID                 `json:"split"`
		CaseProfile artifact.ID                 `json:"case_profile"`
		Scorer      artifact.ID                 `json:"scorer"`
		Execution   artifact.ID                 `json:"execution"`
		Environment artifact.ID                 `json:"environment"`
		Agent       *agentTrajectoryAuthorities `json:"agent,omitempty"`
	}{p.body.Version, p.body.Dataset, p.body.Split, p.body.CaseProfile, p.body.Scorer, p.body.Execution, p.body.Environment, p.body.Agent})
}

// ReadExecutionPolicy decodes a bound execution-policy authority.
func ReadExecutionPolicy(ctx context.Context, reader artifact.Reader, id artifact.ID) (ExecutionPolicy, error) {
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil {
		return ExecutionPolicy{}, err
	}
	if !found {
		return ExecutionPolicy{}, errors.New("evaluation: execution policy is absent")
	}
	if err := executionContract.ValidateContent(content, id); err != nil {
		return ExecutionPolicy{}, err
	}
	var policy ExecutionPolicy
	if err := strictjson.DecodeBytes(content.Data, &policy); err != nil {
		return ExecutionPolicy{}, err
	}
	if !policy.valid() {
		return ExecutionPolicy{}, errors.New("evaluation: invalid execution policy")
	}
	return policy, nil
}

func ParsePlan(content []byte) (Plan, error) {
	var body planBody
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return Plan{}, err
	}
	identity, err := artifact.JSONID(artifact.KindProfile, body)
	if err != nil {
		return Plan{}, err
	}
	plan := Plan{identity: identity, body: body}
	if err := plan.ValidateIdentity(); err != nil {
		return Plan{}, err
	}
	return plan, nil
}

func (p Plan) ValidateIdentity() error {
	if p.body.Version != artifact.InitialDocumentVersion ||
		p.body.ModelDefinition.Kind() != artifact.KindModelDefinition ||
		p.body.RuntimeRecipe.Kind() != artifact.KindRecipe ||
		p.body.Dataset.Kind() != artifact.KindDataset || p.body.Split.Kind() != artifact.KindDatasetShard ||
		p.body.CaseProfile.Kind() != artifact.KindProfile || p.body.Scorer.Kind() != artifact.KindProfile ||
		p.body.Execution.Kind() != artifact.KindProfile || p.body.Environment.Kind() != artifact.KindEvidence ||
		!validCommit(p.body.CodeCommit) {
		return errors.New("evaluation: invalid plan identity authorities")
	}
	if p.body.Agent != nil {
		copy := *p.body.Agent
		if err := canonicalizeAgentTrajectoryAuthorities(&copy); err != nil || !slices.Equal(copy.Trajectories, p.body.Agent.Trajectories) ||
			!slices.Equal(copy.Deterministic, p.body.Agent.Deterministic) || !slices.Equal(copy.AdvisoryJudges, p.body.Agent.AdvisoryJudges) {
			return errors.Join(err, errors.New("evaluation: agent trajectory authorities are not canonical"))
		}
	}
	want, err := artifact.JSONID(artifact.KindProfile, p.body)
	if err != nil || want != p.identity {
		return errors.Join(err, errors.New("evaluation: plan identity differs"))
	}
	return nil
}

func (p Plan) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		p.body.ModelDefinition, p.body.RuntimeRecipe, p.body.Dataset, p.body.Split,
		p.body.CaseProfile, p.body.Scorer, p.body.Execution, p.body.Environment,
	}
	if p.body.Agent != nil {
		parents = append(parents, p.body.Agent.Trajectories...)
		parents = append(parents, p.body.Agent.AdvisoryJudges...)
	}
	return artifact.DependencyLineage(p.identity, parents...)
}

func canonicalizeAgentTrajectoryAuthorities(agent *agentTrajectoryAuthorities) error {
	if agent == nil || len(agent.Trajectories) == 0 || !slices.Equal(agent.Deterministic, requiredAgentTrajectoryScores) {
		return errors.New("evaluation: incomplete deterministic agent trajectory plan")
	}
	for _, group := range []struct {
		values *[]artifact.ID
		kind   artifact.Kind
	}{{&agent.Trajectories, artifact.KindEvidence}, {&agent.AdvisoryJudges, artifact.KindProfile}} {
		for _, id := range *group.values {
			if id.Kind() != group.kind {
				return errors.New("evaluation: invalid agent trajectory authority")
			}
		}
		sort.Slice(*group.values, func(i, j int) bool { return artifact.CompareID((*group.values)[i], (*group.values)[j]) < 0 })
		*group.values = slices.Compact(*group.values)
	}
	return nil
}

// Content returns the native OvergoDB document for external adapter publication.
func (p Plan) Content() (artifact.Content, error) {
	if err := p.ValidateIdentity(); err != nil {
		return artifact.Content{}, err
	}
	return evaluationPlanContract.ContentJSON(p.identity, p.body)
}

func (p Plan) authorityContents() []artifact.Content {
	return slices.Clone(p.authorities)
}

func authorityContent(contract artifact.DocumentContract, value any) (artifact.Content, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return artifact.Content{}, err
	}
	return contract.ContentBytes(data)
}

func validCommit(value string) bool {
	return gitauthority.ValidObjectID(value)
}
