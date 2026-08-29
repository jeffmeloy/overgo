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
	// AgentTaskContractMediaType identifies agent task contracts.
	AgentTaskContractMediaType = "application/vnd.overgo.agent-task-contract+json"
	// AgentTaskContractSchema identifies the agent task contract schema.
	AgentTaskContractSchema = "overgo/agent-task-contract/v1"
)

// AcceptanceCriterion is a deterministic completion obligation, not a model
// completion claim. Verifier names an immutable recipe or evidence authority.
type AcceptanceCriterion struct {
	Name     string      `json:"name"`
	Scope    string      `json:"scope"`
	Verifier artifact.ID `json:"verifier"`
}

// AgentTaskContract binds reusable worker authority to one exact assignment.
type AgentTaskContract struct {
	Version         uint16                `json:"version"`
	ID              artifact.ID           `json:"-"`
	Agent           artifact.ID           `json:"agent"`
	Objective       string                `json:"objective"`
	Scope           []string              `json:"scope"`
	NonGoals        []string              `json:"non_goals,omitempty"`
	AllowedEffects  []string              `json:"allowed_effects"`
	Acceptance      []AcceptanceCriterion `json:"acceptance"`
	Verification    []artifact.ID         `json:"verification"`
	Budget          artifact.ID           `json:"budget"`
	PauseConditions []string              `json:"pause_conditions"`
}

var agentTaskContractCodec = artifact.JSONDocumentCodec(
	"agent task contract", artifact.KindRecipe, AgentTaskContractMediaType, AgentTaskContractSchema,
	canonicalizeAgentTaskContract,
	func(value AgentTaskContract) artifact.ID { return value.ID },
	func(value *AgentTaskContract, id artifact.ID) { value.ID = id },
	func(value AgentTaskContract) AgentTaskContract {
		value.Scope = slices.Clone(value.Scope)
		value.NonGoals = slices.Clone(value.NonGoals)
		value.AllowedEffects = slices.Clone(value.AllowedEffects)
		value.Acceptance = slices.Clone(value.Acceptance)
		value.Verification = slices.Clone(value.Verification)
		value.PauseConditions = slices.Clone(value.PauseConditions)
		return value
	},
)

// NewAgentTaskContract canonicalizes and identifies one immutable task contract.
func NewAgentTaskContract(value AgentTaskContract) (AgentTaskContract, error) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return agentTaskContractCodec.New(value)
}

// RequireAgentTaskContract loads one canonical task contract.
func RequireAgentTaskContract(ctx context.Context, reader artifact.Reader, id artifact.ID) (AgentTaskContract, error) {
	return agentTaskContractCodec.Require(ctx, reader, id)
}

// Content returns the canonical committed bytes of the contract.
func (value AgentTaskContract) Content() (artifact.Content, error) {
	return agentTaskContractCodec.Content(value)
}

// ValidateIdentity checks the contract's canonical form and content-addressed identity.
func (value AgentTaskContract) ValidateIdentity() error {
	return agentTaskContractCodec.ValidateIdentity(value)
}

// Lineage links the contract to its agent, budget, verification authorities,
// and acceptance verifiers.
func (value AgentTaskContract) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Agent, value.Budget}
	parents = append(parents, value.Verification...)
	for _, criterion := range value.Acceptance {
		parents = append(parents, criterion.Verifier)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

func canonicalizeAgentTaskContract(value *AgentTaskContract) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Agent.Kind() != artifact.KindRecipe ||
		value.Budget.Kind() != artifact.KindEvidence || !boundedStatement(value.Objective) || len(value.Scope) == 0 ||
		len(value.AllowedEffects) == 0 || len(value.Acceptance) == 0 || len(value.PauseConditions) == 0 {
		return errors.New("recipe: invalid agent task contract")
	}
	for _, group := range [][]string{value.Scope, value.NonGoals, value.AllowedEffects, value.PauseConditions} {
		for _, item := range group {
			if !boundedStatement(item) {
				return errors.New("recipe: invalid agent task statement")
			}
		}
	}
	slices.Sort(value.Scope)
	value.Scope = slices.Compact(value.Scope)
	slices.Sort(value.NonGoals)
	value.NonGoals = slices.Compact(value.NonGoals)
	slices.Sort(value.AllowedEffects)
	value.AllowedEffects = slices.Compact(value.AllowedEffects)
	slices.Sort(value.PauseConditions)
	value.PauseConditions = slices.Compact(value.PauseConditions)
	for _, criterion := range value.Acceptance {
		if !textcheck.LowerIdentifier(criterion.Name, len(criterion.Name)) || !boundedStatement(criterion.Scope) || !criterion.Verifier.Valid() {
			return errors.New("recipe: invalid acceptance criterion")
		}
	}
	sort.Slice(value.Acceptance, func(i, j int) bool { return value.Acceptance[i].Name < value.Acceptance[j].Name })
	for _, id := range value.Verification {
		if !id.Valid() {
			return errors.New("recipe: invalid verification authority")
		}
	}
	sort.Slice(value.Verification, func(i, j int) bool { return artifact.CompareID(value.Verification[i], value.Verification[j]) < 0 })
	value.Verification = slices.Compact(value.Verification)
	return nil
}

func boundedStatement(value string) bool {
	return value != "" && textcheck.Bounded(value, len(value), "\x00\r\n")
}
