package agentloop

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const projectAgentScope = "*"

// ContractState is the deterministic runtime view of one immutable task
// contract. Durable obligations remain owned by runrecord.
type ContractState struct {
	mu          sync.Mutex
	contract    recipe.AgentTaskContract
	obligations []runrecord.AgentObligation
	epochs      map[string]uint64
}

// NewContractState validates the contract identity and binds only obligations
// belonging to that task into a fresh runtime view.
func NewContractState(contract recipe.AgentTaskContract, obligations []runrecord.AgentObligation) (*ContractState, error) {
	if err := contract.ValidateIdentity(); err != nil {
		return nil, err
	}
	for _, obligation := range obligations {
		if obligation.Task != contract.ID {
			return nil, errors.New("agent loop: obligation belongs to another task")
		}
	}
	return &ContractState{contract: contract, obligations: slices.Clone(obligations), epochs: map[string]uint64{}}, nil
}

// Authority returns the immutable task ceiling this runtime view enforces.
func (state *ContractState) Authority() artifact.ID {
	if state == nil {
		return artifact.ID{}
	}
	return state.contract.ID
}

// Admit refuses mutations outside the contract's exact effect ceiling.
func (state *ContractState) Admit(effect agenttool.InvocationEffect) error {
	if state == nil || effect.Class != agenttool.EffectMutation {
		return nil
	}
	if effect.OpaqueMutation || !effect.Known {
		return errors.New("agent loop: task contract refuses opaque mutation")
	}
	for _, target := range effect.Targets {
		grant := string(target.Scope) + ":" + string(effect.Class)
		if !slices.Contains(state.contract.AllowedEffects, grant) && !slices.Contains(state.contract.AllowedEffects, string(effect.Class)) {
			return fmt.Errorf("agent loop: task contract does not grant %s", grant)
		}
	}
	return nil
}

// ObserveMutation advances only scopes touched by the exact effect. An opaque
// effect advances project scope and therefore invalidates every obligation.
func (state *ContractState) ObserveMutation(effect agenttool.InvocationEffect) {
	if state == nil || effect.Class != agenttool.EffectMutation {
		return
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if effect.OpaqueMutation || !effect.Known {
		state.epochs[projectAgentScope]++
		return
	}
	for _, obligation := range state.obligations {
		if slices.ContainsFunc(effect.Targets, func(target agenttool.InvocationTarget) bool {
			return scopesOverlap(target.Value, obligation.Scope)
		}) {
			state.epochs[obligation.Scope]++
		}
	}
}

func scopesOverlap(left, right string) bool {
	left, right = strings.TrimSuffix(left, "/"), strings.TrimSuffix(right, "/")
	return left == right || strings.HasPrefix(left, right+"/") || strings.HasPrefix(right, left+"/")
}
