package evaluation

import (
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"

	"overgo/internal/artifact"
)

const (
	evaluationPlanVersion   uint16 = 1
	evaluationPlanMediaType        = "application/vnd.overgo.evaluation-plan+json"
	evaluationPlanSchema           = "overgo/evaluation-plan/v1"
)

var evaluationPlanContract = artifact.DocumentContract{
	Kind: artifact.KindProfile, MediaType: evaluationPlanMediaType, Schema: evaluationPlanSchema,
}

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
	identity artifact.ID
	body     planBody
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
	scorer, err := artifact.JSONID(artifact.KindProfile, struct {
		Version uint16 `json:"version"`
		Kind    string `json:"kind"`
	}{Version: evaluationPlanVersion, Kind: "exact-generation"})
	if err != nil {
		return Plan{}, err
	}
	if authorities.Execution.Lifecycle != LifecycleIsolated && authorities.Execution.Lifecycle != LifecycleResident {
		return Plan{}, errors.New("evaluation: invalid execution policy")
	}
	execution, err := artifact.JSONID(artifact.KindProfile, authorities.Execution)
	if err != nil {
		return Plan{}, err
	}
	body := planBody{
		Version:         evaluationPlanVersion,
		ModelDefinition: authorities.ModelDefinition, RuntimeRecipe: authorities.RuntimeRecipe,
		Dataset: exact.dataset, Split: exact.split, CaseProfile: exact.identity,
		Scorer: scorer, Execution: execution, CodeCommit: authorities.CodeCommit,
		Environment: authorities.Environment,
	}
	identity, err := artifact.JSONID(artifact.KindProfile, body)
	if err != nil {
		return Plan{}, err
	}
	return Plan{identity: identity, body: body}, nil
}

func (p Plan) Identity() artifact.ID { return p.identity }

func (p Plan) content() (artifact.Content, error) {
	data, err := json.Marshal(p.body)
	if err != nil {
		return artifact.Content{}, err
	}
	return evaluationPlanContract.Content(p.identity, data)
}

func validCommit(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && (len(decoded) == sha1.Size || len(decoded) == sha256.Size)
}
