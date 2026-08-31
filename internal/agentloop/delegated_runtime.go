package agentloop

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
)

// DelegatedCapabilityRuntime joins the staged delegation owners into one
// production dispatch path: the compiled grant bounds the manuals, the
// delegated coordinator proposes under the invocation's step budget and
// ceiling, the capability proxy dispatches only what the grant admits,
// mutation checkpointing supervises restores, and the task contract's
// acceptance criteria become open obligations that only exact-scope
// evidence at the current mutation epoch can resolve.
type DelegatedCapabilityRuntime struct {
	Compiled    workflowruntime.CompiledDelegation
	Coordinator *DelegatedCoordinator
	Checkpoints *MutationCheckpointRuntime
	Task        recipe.AgentTaskContract
	Obligations []runrecord.AgentObligation
}

// initialMutationEpoch is the epoch obligations open at: the first mutation
// the runtime supervises advances it, invalidating stale evidence.
const initialMutationEpoch = 1

// NewDelegatedCapabilityRuntime compiles the invocation against the worker
// ceiling and active catalog, wires the delegated coordinator and mutation
// checkpoint runtime, registers the capability proxy behind the grant's own
// effect admission, and derives one obligation per acceptance criterion at
// the initial mutation epoch. It grants nothing the invocation did not
// already carry.
func NewDelegatedCapabilityRuntime(
	ctx context.Context,
	store artifact.Repository,
	executor *agenttool.Executor,
	invocation recipe.DelegatedAgentInvocation,
	worker recipe.AgentDefinition,
	task recipe.AgentTaskContract,
	node recipe.NodeID,
	workspaceRoot string,
) (*DelegatedCapabilityRuntime, error) {
	if invocation.Task != task.ID {
		return nil, errors.New("agent loop: delegated invocation names a different task contract")
	}
	compiled, err := workflowruntime.CompileDelegatedAgentInvocation(ctx, store, invocation, worker)
	if err != nil {
		return nil, err
	}
	coordinator, err := NewDelegatedCoordinator(store, executor, invocation, node)
	if err != nil {
		return nil, err
	}
	if err := agenttool.RegisterCapabilityProxy(executor, store, GrantEffectAdmission(invocation.Grant)); err != nil {
		return nil, err
	}
	checkpoints, err := NewMutationCheckpointRuntime(workspaceRoot, store)
	if err != nil {
		return nil, err
	}
	obligations := make([]runrecord.AgentObligation, 0, len(task.Acceptance))
	for _, criterion := range task.Acceptance {
		obligation, obligationErr := runrecord.NewAgentObligation(runrecord.AgentObligation{
			Task: task.ID, Name: criterion.Name, Scope: criterion.Scope,
			MutationEpoch: initialMutationEpoch,
			Sources:       []artifact.ID{criterion.Verifier},
		})
		if obligationErr != nil {
			return nil, obligationErr
		}
		obligations = append(obligations, obligation)
	}
	return &DelegatedCapabilityRuntime{
		Compiled: compiled, Coordinator: coordinator, Checkpoints: checkpoints,
		Task: task, Obligations: obligations,
	}, nil
}

// SessionManuals returns the delegated session's visible manual set: the
// compiled grant plus the capability proxy manual the runtime registered,
// so a delegated session reaches dynamic capabilities only through the
// grant-admitted proxy.
func (runtime *DelegatedCapabilityRuntime) SessionManuals() ([]agenttool.Manual, error) {
	if runtime == nil {
		return nil, errors.New("agent loop: delegated runtime is absent")
	}
	proxy, err := agenttool.CapabilityProxyManual()
	if err != nil {
		return nil, err
	}
	return append(slices.Clone(runtime.Compiled.Manuals), proxy), nil
}

// GrantEffectAdmission derives the capability proxy's admission policy from
// the delegated grant alone: a read-only grant admits only inspections, and
// every admitted class must be granted explicitly. The policy adds no
// authority -- it can only narrow what the proxy dispatches.
func GrantEffectAdmission(grant recipe.AgentCapabilityGrant) agenttool.EffectAdmission {
	allowed := slices.Clone(grant.AllowedEffects)
	readOnly := grant.ReadOnly
	return func(effect agenttool.InvocationEffect) error {
		if readOnly && effect.Class != agenttool.EffectInspection {
			return fmt.Errorf("agent loop: read-only grant refuses %s dispatch", effect.Class)
		}
		if !slices.Contains(allowed, string(effect.Class)) {
			return fmt.Errorf("agent loop: grant does not admit %s effects", effect.Class)
		}
		return nil
	}
}

// ResolveObligation records exact-scope evidence against one open
// obligation and reports whether the full acceptance set is now satisfied
// at the resolved mutation epoch. Absence of a resolution stays open;
// nothing here infers success.
func (runtime *DelegatedCapabilityRuntime) ResolveObligation(
	name string,
	epoch uint64,
	evidence []artifact.ID,
	resolved []runrecord.AgentObligationResolution,
) ([]runrecord.AgentObligationResolution, bool, error) {
	if runtime == nil {
		return nil, false, errors.New("agent loop: delegated runtime is absent")
	}
	for _, obligation := range runtime.Obligations {
		if obligation.Name != name {
			continue
		}
		resolution, err := runrecord.NewAgentObligationResolution(runrecord.AgentObligationResolution{
			Obligation: obligation.ID, Scope: obligation.Scope,
			MutationEpoch: epoch, Evidence: evidence,
		})
		if err != nil {
			return nil, false, err
		}
		resolved = append(resolved, resolution)
		for _, remaining := range runtime.Obligations {
			if !runrecord.AgentObligationSatisfied(remaining, resolved) {
				return resolved, false, nil
			}
		}
		return resolved, true, nil
	}
	return nil, false, fmt.Errorf("agent loop: task contract has no obligation %q", name)
}
