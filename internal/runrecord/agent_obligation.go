package runrecord

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// AgentObligationMediaType identifies agent obligations.
	AgentObligationMediaType = "application/vnd.overgo.agent-obligation+json"
	// AgentObligationSchema identifies the agent obligation schema.
	AgentObligationSchema = "overgo/agent-obligation/v1"
	// AgentObligationResolutionMediaType identifies obligation resolutions.
	AgentObligationResolutionMediaType = "application/vnd.overgo.agent-obligation-resolution+json"
	// AgentObligationResolutionSchema identifies the obligation resolution schema.
	AgentObligationResolutionSchema = "overgo/agent-obligation-resolution/v1"
)

// AgentObligation is one requirement derived from committed authorities.
type AgentObligation struct {
	Version       uint16        `json:"version"`
	ID            artifact.ID   `json:"-"`
	Task          artifact.ID   `json:"task"`
	Name          string        `json:"name"`
	Scope         string        `json:"scope"`
	MutationEpoch uint64        `json:"mutation_epoch"`
	Sources       []artifact.ID `json:"sources"`
}

// AgentObligationResolution binds a requirement to exact evidence. Absence of
// this document means open; there is no inferred success state.
type AgentObligationResolution struct {
	Version       uint16        `json:"version"`
	ID            artifact.ID   `json:"-"`
	Obligation    artifact.ID   `json:"obligation"`
	Scope         string        `json:"scope"`
	MutationEpoch uint64        `json:"mutation_epoch"`
	Evidence      []artifact.ID `json:"evidence"`
}

var agentObligationCodec = artifact.JSONDocumentCodec(
	"agent obligation", artifact.KindEvidence, AgentObligationMediaType, AgentObligationSchema,
	canonicalizeAgentObligation, func(v AgentObligation) artifact.ID { return v.ID },
	func(v *AgentObligation, id artifact.ID) { v.ID = id },
	func(v AgentObligation) AgentObligation { v.Sources = slices.Clone(v.Sources); return v },
)
var agentObligationResolutionCodec = artifact.JSONDocumentCodec(
	"agent obligation resolution", artifact.KindEvidence, AgentObligationResolutionMediaType, AgentObligationResolutionSchema,
	canonicalizeAgentObligationResolution, func(v AgentObligationResolution) artifact.ID { return v.ID },
	func(v *AgentObligationResolution, id artifact.ID) { v.ID = id },
	func(v AgentObligationResolution) AgentObligationResolution {
		v.Evidence = slices.Clone(v.Evidence)
		return v
	},
)

// NewAgentObligation canonicalizes and identifies one immutable obligation.
func NewAgentObligation(value AgentObligation) (AgentObligation, error) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return agentObligationCodec.New(value)
}

// NewAgentObligationResolution canonicalizes and identifies one immutable resolution.
func NewAgentObligationResolution(value AgentObligationResolution) (AgentObligationResolution, error) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return agentObligationResolutionCodec.New(value)
}

// Content returns the canonical committed bytes of the obligation.
func (v AgentObligation) Content() (artifact.Content, error) { return agentObligationCodec.Content(v) }

// Content returns the canonical committed bytes of the resolution.
func (v AgentObligationResolution) Content() (artifact.Content, error) {
	return agentObligationResolutionCodec.Content(v)
}

// Lineage links the obligation to its task and sources.
func (v AgentObligation) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(v.ID, append([]artifact.ID{v.Task}, v.Sources...)...)
}

// Lineage links the resolution to its obligation and evidence.
func (v AgentObligationResolution) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(v.ID, append([]artifact.ID{v.Obligation}, v.Evidence...)...)
}

// AgentObligationSatisfied requires exact scope and epoch agreement.
func AgentObligationSatisfied(obligation AgentObligation, resolutions []AgentObligationResolution) bool {
	return slices.ContainsFunc(resolutions, func(resolution AgentObligationResolution) bool {
		return resolution.Obligation == obligation.ID && resolution.Scope == obligation.Scope &&
			resolution.MutationEpoch == obligation.MutationEpoch && len(resolution.Evidence) != 0
	})
}

func canonicalizeAgentObligation(v *AgentObligation) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || v.Task.Kind() != artifact.KindRecipe ||
		!textcheck.LowerIdentifier(v.Name, len(v.Name)) || !validAgentScope(v.Scope) || len(v.Sources) == 0 {
		return errors.New("run record: invalid agent obligation")
	}
	for _, id := range v.Sources {
		if !id.Valid() {
			return errors.New("run record: invalid obligation source")
		}
	}
	sort.Slice(v.Sources, func(i, j int) bool { return artifact.CompareID(v.Sources[i], v.Sources[j]) < 0 })
	v.Sources = slices.Compact(v.Sources)
	return nil
}
func canonicalizeAgentObligationResolution(v *AgentObligationResolution) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || v.Obligation.Kind() != artifact.KindEvidence ||
		!validAgentScope(v.Scope) || len(v.Evidence) == 0 {
		return errors.New("run record: invalid agent obligation resolution")
	}
	for _, id := range v.Evidence {
		if !id.Valid() {
			return errors.New("run record: invalid obligation evidence")
		}
	}
	sort.Slice(v.Evidence, func(i, j int) bool { return artifact.CompareID(v.Evidence[i], v.Evidence[j]) < 0 })
	v.Evidence = slices.Compact(v.Evidence)
	return nil
}
func validAgentScope(scope string) bool {
	return scope != "" && textcheck.Bounded(scope, len(scope), "\x00\r\n")
}
