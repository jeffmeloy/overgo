package evaluation

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"slices"

	"overgo/internal/artifact"
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
	Version         uint16      `json:"version"`
	ModelDefinition artifact.ID `json:"model_definition"`
	RuntimeRecipe   artifact.ID `json:"runtime_recipe"`
	Dataset         artifact.ID `json:"dataset"`
	Split           artifact.ID `json:"split"`
	CaseProfile     artifact.ID `json:"case_profile"`
	Scorer          artifact.ID `json:"scorer"`
	Execution       artifact.ID `json:"execution"`
	CodeCommit      string      `json:"code_commit"`
	Environment     artifact.ID `json:"environment"`
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
	if authorities.Execution.Lifecycle != LifecycleIsolated && authorities.Execution.Lifecycle != LifecycleResident {
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
	want, err := artifact.JSONID(artifact.KindProfile, p.body)
	if err != nil || want != p.identity {
		return errors.Join(err, errors.New("evaluation: plan identity differs"))
	}
	return nil
}

func (p Plan) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(p.identity,
		p.body.ModelDefinition, p.body.RuntimeRecipe, p.body.Dataset, p.body.Split,
		p.body.CaseProfile, p.body.Scorer, p.body.Execution, p.body.Environment,
	)
}

// Content returns the native RepoDB document for external adapter publication.
func (p Plan) Content() (artifact.Content, error) {
	if err := p.ValidateIdentity(); err != nil {
		return artifact.Content{}, err
	}
	data, err := json.Marshal(p.body)
	if err != nil {
		return artifact.Content{}, err
	}
	return evaluationPlanContract.Content(p.identity, data)
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
	decoded, err := hex.DecodeString(value)
	return err == nil && (len(decoded) == sha1.Size || len(decoded) == sha256.Size)
}
