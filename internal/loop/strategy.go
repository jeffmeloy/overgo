package loop

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// StrategyEnvironment carries the human-readable strategy label from a
// driver to gate attempts. Content-addressed strategy authority is recorded
// separately by Strategy.ID.
const StrategyEnvironment = "OVERGO_STRATEGY"

const strategyDigestLength = 12

// StrategyIdentity returns an explicit bounded label or derives one from the
// exact worker command. It is compatibility metadata, not execution authority.
func StrategyIdentity(worker []string, declared string) string {
	if name := strings.TrimSpace(declared); name != "" {
		return name
	}
	if len(worker) == 0 {
		return ""
	}
	digest := sha256.Sum256([]byte(strings.Join(worker, "\x00")))
	return "worker-" + fmt.Sprintf("%x", digest)[:strategyDigestLength]
}

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
	func(v Strategy) Strategy {
		v.Policies = slices.Clone(v.Policies)
		if v.Loop.Closure != nil {
			closure := *v.Loop.Closure
			v.Loop.Closure = &closure
		}
		return v
	},
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
	record.StrategyID = v.ID
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
	var noProposals uint64
	if v.Loop.Closure != nil && closureStopReason(*v.Loop.Closure, ClosureFacts{MeasuredGain: true}, noProposals) == ReasonBudget {
		return errors.New("loop: invalid strategy closure budget")
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
