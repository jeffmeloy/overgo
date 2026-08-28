package workflowruntime

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// CompiledDelegation is the executable projection of exact recipe authority.
type CompiledDelegation struct {
	Invocation recipe.DelegatedAgentInvocation
	Manuals    []agenttool.Manual
}

// CompileDelegatedAgentInvocation loads every granted manual and validates it
// against the worker ceiling, exact catalog membership, and the read-only bound.
func CompileDelegatedAgentInvocation(ctx context.Context, reader artifact.Reader, invocation recipe.DelegatedAgentInvocation, worker recipe.AgentDefinition) (CompiledDelegation, error) {
	if ctx == nil || reader == nil {
		return CompiledDelegation{}, errors.New("workflow runtime: delegation authority is absent")
	}
	if err := invocation.ValidateIdentity(); err != nil {
		return CompiledDelegation{}, err
	}
	if err := worker.ValidateIdentity(); err != nil {
		return CompiledDelegation{}, err
	}
	if invocation.Worker != worker.ID {
		return CompiledDelegation{}, errors.New("workflow runtime: delegation binds another worker")
	}
	snapshot, err := agenttool.RequireCatalogSnapshot(ctx, reader, invocation.Catalog)
	if err != nil {
		return CompiledDelegation{}, err
	}
	manuals := make([]agenttool.Manual, 0, len(invocation.Grant.Manuals))
	for _, id := range invocation.Grant.Manuals {
		if !slices.Contains(worker.ToolManuals, id) {
			return CompiledDelegation{}, errors.New("workflow runtime: delegated manual exceeds worker ceiling")
		}
		entryFound := slices.ContainsFunc(snapshot.Entries, func(entry agenttool.CatalogEntry) bool { return entry.Manual == id })
		if !entryFound {
			return CompiledDelegation{}, errors.New("workflow runtime: delegated manual is outside exact catalog")
		}
		manual, err := agenttool.LoadManual(ctx, reader, id)
		if err != nil {
			return CompiledDelegation{}, err
		}
		if invocation.Grant.ReadOnly && manual.Effect != agenttool.EffectInspection {
			return CompiledDelegation{}, errors.New("workflow runtime: read-only child includes mutation manual")
		}
		manuals = append(manuals, manual)
	}
	return CompiledDelegation{Invocation: invocation, Manuals: manuals}, nil
}

// DeriveCausal binds the delegated execution into the delegating
// execution's causal chain: the child runs under the same causal root,
// with the delegating execution as its causal subject. Authority is
// unaffected -- it comes from the compiled grant, never from the chain.
func (compiled CompiledDelegation) DeriveCausal(parent runrecord.CausalContext, delegator artifact.ID) (runrecord.CausalContext, error) {
	return parent.Derive(runrecord.TriggerDelegation, delegator)
}

// ManualIDs returns the compiled manuals' identities in grant order.
func (compiled CompiledDelegation) ManualIDs() []artifact.ID {
	result := make([]artifact.ID, len(compiled.Manuals))
	for index, manual := range compiled.Manuals {
		result[index] = manual.ID
	}
	return result
}
