package loop

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const (
	StrategyMediaType = "application/vnd.overgo.agent-strategy+json"
	StrategySchema    = "overgo/agent-strategy/v1"
)

type Strategy struct {
	Version     uint16        `json:"version"`
	ID          artifact.ID   `json:"-"`
	Worker      artifact.ID   `json:"worker"`
	Prompt      artifact.ID   `json:"prompt"`
	ModelRecipe artifact.ID   `json:"model_recipe"`
	Catalog     artifact.ID   `json:"catalog"`
	Policies    []artifact.ID `json:"policies"`
	Loop        Config        `json:"loop"`
}

var strategyCodec = artifact.JSONDocumentCodec(
	"agent strategy", artifact.KindProfile, StrategyMediaType, StrategySchema,
	canonicalizeStrategy, func(v Strategy) artifact.ID { return v.ID }, func(v *Strategy, id artifact.ID) { v.ID = id },
	func(v Strategy) Strategy { v.Policies = slices.Clone(v.Policies); return v },
)

func NewStrategy(worker recipe.AgentDefinition, catalog artifact.ID, config Config) (Strategy, error) {
	if err := worker.ValidateIdentity(); err != nil {
		return Strategy{}, err
	}
	return strategyCodec.New(Strategy{Version: artifact.InitialDocumentVersion, Worker: worker.ID, Prompt: worker.Prompt,
		ModelRecipe: worker.ModelRecipe, Catalog: catalog, Policies: slices.Clone(worker.Policies), Loop: config})
}
func (v Strategy) Content() (artifact.Content, error) { return strategyCodec.Content(v) }
func (v Strategy) ValidateIdentity() error            { return strategyCodec.ValidateIdentity(v) }
func (v Strategy) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(v.ID, append([]artifact.ID{v.Worker, v.Prompt, v.ModelRecipe, v.Catalog}, v.Policies...)...)
}

func (v Strategy) StampAttempt(record runrecord.AttemptRecord) (runrecord.AttemptRecord, error) {
	if err := v.ValidateIdentity(); err != nil {
		return runrecord.AttemptRecord{}, err
	}
	record.Strategy = v.ID
	return runrecord.NewAttemptRecord(record)
}
func (v Strategy) StampTrajectory(trace runrecord.InteractionTrace) (runrecord.InteractionTrace, error) {
	if err := v.ValidateIdentity(); err != nil {
		return runrecord.InteractionTrace{}, err
	}
	trace.Strategy = v.ID
	return runrecord.NewAgentTrajectory(trace)
}

func canonicalizeStrategy(v *Strategy) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || v.Worker.Kind() != artifact.KindRecipe || v.Prompt.Kind() != artifact.KindFile ||
		v.ModelRecipe.Kind() != artifact.KindRecipe || v.Catalog.Kind() != artifact.KindProfile || len(v.Policies) == 0 || v.Loop.MaxAttemptsPerStep <= 0 || v.Loop.MaxInvocations <= 0 {
		return errors.New("loop: invalid agent strategy")
	}
	for _, id := range v.Policies {
		if id.Kind() != artifact.KindProfile {
			return errors.New("loop: invalid strategy policy")
		}
	}
	sort.Slice(v.Policies, func(i, j int) bool { return artifact.CompareID(v.Policies[i], v.Policies[j]) < 0 })
	v.Policies = slices.Compact(v.Policies)
	return nil
}
