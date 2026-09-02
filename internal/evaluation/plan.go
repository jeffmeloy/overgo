package evaluation

import (
	"context"
	"encoding/json"
	"errors"
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

type Lifecycle string

const (
	LifecycleIsolated Lifecycle = "isolated"
	LifecycleResident Lifecycle = "resident"
)

type ExecutionPolicy struct {
	Lifecycle Lifecycle `json:"lifecycle"`
	// Prompting names how a scored prompt reaches the model: the raw
	// completion text (the recorded anchor, the empty default so
	// earlier plans keep their identity) or the model's own declared
	// chat template. Two protocols are two plans, so their records
	// never collapse into one "latest" score.
	Prompting Prompting `json:"prompting,omitzero"`
}

// Prompting is the prompt-shaping protocol an execution policy binds.
type Prompting string

const (
	// PromptingRawCompletion scores the suite's prompt text unchanged.
	PromptingRawCompletion Prompting = ""
	// PromptingChatTemplate shapes each prompt through the model's
	// declared chat template before scoring.
	PromptingChatTemplate Prompting = "chat-template"
)

// Label names the protocol for reports; the raw default reads as such.
func (prompting Prompting) Label() string {
	if prompting == PromptingRawCompletion {
		return "raw-completion"
	}
	return string(prompting)
}

type ExactAuthorities struct {
	ModelDefinition artifact.ID
	RuntimeRecipe   artifact.ID
	CodeCommit      string
	Environment     artifact.ID
	Execution       ExecutionPolicy
}

type Plan struct {
	identity    artifact.ID
	body        planBody
	authorities []artifact.Content
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
	Agent           *AgentTrajectoryAuthorities `json:"agent,omitempty"`
}

// AgentTrajectoryScore names one deterministic, code-owned trajectory scorer.
type AgentTrajectoryScore string

const (
	// AgentScoreTaskSuccess names the scorer for task acceptance success.
	AgentScoreTaskSuccess AgentTrajectoryScore = "task-success"
	// AgentScoreEvidenceCompleteness names the scorer for committed-evidence completeness.
	AgentScoreEvidenceCompleteness AgentTrajectoryScore = "evidence-completeness"
	// AgentScoreGateAccuracy names the scorer for gate-result accuracy.
	AgentScoreGateAccuracy AgentTrajectoryScore = "gate-accuracy"
	// AgentScoreEffectPrediction names the scorer for predicted-versus-observed effects.
	AgentScoreEffectPrediction AgentTrajectoryScore = "effect-prediction"
	// AgentScoreRegressions names the scorer for introduced regressions.
	AgentScoreRegressions AgentTrajectoryScore = "regressions"
	// AgentScoreChurn names the scorer for code churn spent on the task.
	AgentScoreChurn AgentTrajectoryScore = "churn"
	// AgentScoreRepeatedDirections names the scorer for repeated attempt directions.
	AgentScoreRepeatedDirections AgentTrajectoryScore = "repeated-directions"
	// AgentScoreBudget names the scorer for consumed budget.
	AgentScoreBudget AgentTrajectoryScore = "budget"
	// AgentScoreLatency names the scorer for wall-clock latency.
	AgentScoreLatency AgentTrajectoryScore = "latency"
	// AgentScoreUnsupportedCompletion names the scorer for completion claims without evidence.
	AgentScoreUnsupportedCompletion AgentTrajectoryScore = "unsupported-completion"
)

// AgentTrajectoryAuthorities extend the exact plan. Deterministic scorers are
// code-owned hard facts; advisory judges are versioned profiles only.
type AgentTrajectoryAuthorities struct {
	Trajectories   []artifact.ID          `json:"trajectories"`
	Deterministic  []AgentTrajectoryScore `json:"deterministic"`
	AdvisoryJudges []artifact.ID          `json:"advisory_judges,omitempty"`
}

var requiredAgentTrajectoryScores = []AgentTrajectoryScore{
	AgentScoreTaskSuccess, AgentScoreEvidenceCompleteness, AgentScoreGateAccuracy, AgentScoreEffectPrediction,
	AgentScoreRegressions, AgentScoreChurn, AgentScoreRepeatedDirections, AgentScoreBudget, AgentScoreLatency,
	AgentScoreUnsupportedCompletion,
}

// BindAgentTrajectoryPlan binds exact trajectories to the existing evaluation
// plan identity without granting model judges promotion authority.
func BindAgentTrajectoryPlan(base Plan, trajectories, advisoryJudges []artifact.ID) (Plan, error) {
	if err := base.ValidateIdentity(); err != nil {
		return Plan{}, err
	}
	agent := &AgentTrajectoryAuthorities{Trajectories: slices.Clone(trajectories), Deterministic: slices.Clone(requiredAgentTrajectoryScores), AdvisoryJudges: slices.Clone(advisoryJudges)}
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

// AgentTrajectoryAuthorities returns a deep copy of the plan's agent
// trajectory extension; ok is false when the plan carries none.
func (p Plan) AgentTrajectoryAuthorities() (AgentTrajectoryAuthorities, bool) {
	if p.body.Agent == nil {
		return AgentTrajectoryAuthorities{}, false
	}
	copy := *p.body.Agent
	copy.Trajectories = slices.Clone(copy.Trajectories)
	copy.Deterministic = slices.Clone(copy.Deterministic)
	copy.AdvisoryJudges = slices.Clone(copy.AdvisoryJudges)
	return copy, true
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

func bindPlan[C, S any](
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
	if authorities.Execution.Lifecycle != LifecycleIsolated && authorities.Execution.Lifecycle != LifecycleResident ||
		authorities.Execution.Prompting != PromptingRawCompletion && authorities.Execution.Prompting != PromptingChatTemplate {
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
	}, nil
}

func (p Plan) Identity() artifact.ID { return p.identity }

func (p Plan) Dataset() artifact.ID { return p.body.Dataset }

// Execution names the plan's bound execution-policy authority.
func (p Plan) Execution() artifact.ID { return p.body.Execution }

// ReadExecutionPolicy decodes a bound execution-policy authority.
func ReadExecutionPolicy(ctx context.Context, reader artifact.Reader, id artifact.ID) (ExecutionPolicy, error) {
	content, found, err := artifact.ReadContent(ctx, reader, id)
	if err != nil {
		return ExecutionPolicy{}, err
	}
	if !found {
		return ExecutionPolicy{}, errors.New("evaluation: execution policy is absent")
	}
	var policy ExecutionPolicy
	if err := json.Unmarshal(content.Data, &policy); err != nil {
		return ExecutionPolicy{}, err
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

func canonicalizeAgentTrajectoryAuthorities(agent *AgentTrajectoryAuthorities) error {
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
