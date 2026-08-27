package recipe

import (
	"context"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	DelegatedAgentInvocationMediaType = "application/vnd.overgo.delegated-agent-invocation+json"
	DelegatedAgentInvocationSchema    = "overgo/delegated-agent-invocation/v1"
)

type AgentCapabilityGrant struct {
	Manuals        []artifact.ID `json:"manuals"`
	AllowedEffects []string      `json:"allowed_effects"`
	ReadOnly       bool          `json:"read_only,omitempty"`
}

type AgentSchedulerPolicy struct {
	MaxSteps    int  `json:"max_steps"`
	MaxParallel int  `json:"max_parallel"`
	Preemptible bool `json:"preemptible,omitempty"`
}

// DelegatedAgentInvocation keeps assignment, worker, capability, context, and
// scheduling authority distinct and immutable.
type DelegatedAgentInvocation struct {
	Version         uint16               `json:"version"`
	ID              artifact.ID          `json:"-"`
	Task            artifact.ID          `json:"task"`
	Worker          artifact.ID          `json:"worker"`
	Grant           AgentCapabilityGrant `json:"grant"`
	StartingContext []artifact.ID        `json:"starting_context,omitempty"`
	Scheduler       AgentSchedulerPolicy `json:"scheduler"`
	Catalog         artifact.ID          `json:"catalog"`
	Model           artifact.ID          `json:"model"`
}

var delegatedAgentInvocationCodec = artifact.JSONDocumentCodec(
	"delegated agent invocation", artifact.KindRecipe, DelegatedAgentInvocationMediaType, DelegatedAgentInvocationSchema,
	canonicalizeDelegatedAgentInvocation,
	func(v DelegatedAgentInvocation) artifact.ID { return v.ID }, func(v *DelegatedAgentInvocation, id artifact.ID) { v.ID = id },
	func(v DelegatedAgentInvocation) DelegatedAgentInvocation {
		v.Grant.Manuals = slices.Clone(v.Grant.Manuals)
		v.Grant.AllowedEffects = slices.Clone(v.Grant.AllowedEffects)
		v.StartingContext = slices.Clone(v.StartingContext)
		return v
	},
)

func NewDelegatedAgentInvocation(v DelegatedAgentInvocation) (DelegatedAgentInvocation, error) {
	v.Version, v.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return delegatedAgentInvocationCodec.New(v)
}
func RequireDelegatedAgentInvocation(ctx context.Context, reader artifact.Reader, id artifact.ID) (DelegatedAgentInvocation, error) {
	return delegatedAgentInvocationCodec.Require(ctx, reader, id)
}
func (v DelegatedAgentInvocation) Content() (artifact.Content, error) {
	return delegatedAgentInvocationCodec.Content(v)
}
func (v DelegatedAgentInvocation) ValidateIdentity() error {
	return delegatedAgentInvocationCodec.ValidateIdentity(v)
}
func (v DelegatedAgentInvocation) Lineage() []artifact.Lineage {
	parents := []artifact.ID{v.Task, v.Worker, v.Catalog, v.Model}
	parents = append(parents, v.Grant.Manuals...)
	parents = append(parents, v.StartingContext...)
	return artifact.DependencyLineage(v.ID, parents...)
}

func canonicalizeDelegatedAgentInvocation(v *DelegatedAgentInvocation) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || v.Task.Kind() != artifact.KindRecipe ||
		v.Worker.Kind() != artifact.KindRecipe || v.Catalog.Kind() != artifact.KindProfile || v.Model.Kind() != artifact.KindModel ||
		len(v.Grant.Manuals) == 0 || len(v.Grant.AllowedEffects) == 0 || v.Scheduler.MaxSteps <= 0 || v.Scheduler.MaxParallel <= 0 {
		return errors.New("recipe: invalid delegated agent invocation")
	}
	for _, effect := range v.Grant.AllowedEffects {
		if !boundedStatement(effect) {
			return errors.New("recipe: invalid delegated effect grant")
		}
	}
	if v.Grant.ReadOnly && slices.Contains(v.Grant.AllowedEffects, "mutation") {
		return errors.New("recipe: read-only delegation grants mutation")
	}
	for _, group := range []*[]artifact.ID{&v.Grant.Manuals, &v.StartingContext} {
		for _, id := range *group {
			if !id.Valid() {
				return errors.New("recipe: invalid delegated authority")
			}
		}
		sort.Slice(*group, func(i, j int) bool { return artifact.CompareID((*group)[i], (*group)[j]) < 0 })
		*group = slices.Compact(*group)
	}
	sort.Strings(v.Grant.AllowedEffects)
	v.Grant.AllowedEffects = slices.Compact(v.Grant.AllowedEffects)
	if slices.ContainsFunc(v.Grant.AllowedEffects, func(value string) bool { return !textcheck.Bounded(value, len(value), "\x00\r\n") }) {
		return errors.New("recipe: invalid delegated effect")
	}
	return nil
}
