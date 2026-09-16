package agentloop

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/worklease"
)

// SessionRecoveryAuthority binds every mutable and immutable fact that may
// affect the next action. Committed artifacts remain the source of truth.
type SessionRecoveryAuthority struct {
	Task          recipe.AgentTaskContract
	Agent         recipe.AgentDefinition
	Model         artifact.ID
	Catalog       artifact.ID
	Policies      []artifact.ID
	Lease         *worklease.Lease
	MutationEpoch uint64
	Obligations   []runrecord.AgentObligation
	Resolutions   []runrecord.AgentObligationResolution
	Checkpoints   []runrecord.AgentMutationCheckpoint
}

// SessionRecovery is the reconciled restart state: the restored session, the
// next uncommitted action, still-open obligations, and an optional checkpoint
// that must be restored before new work.
type SessionRecovery struct {
	Session           *Session
	NextAction        string
	Outstanding       []runrecord.AgentObligation
	RestoreCheckpoint *artifact.ID
}

// RecoverAgentSession reconciles committed interaction and lifecycle facts. A
// torn executed action is never treated as the next uncommitted action.
func (c *Coordinator) RecoverAgentSession(ctx context.Context, sessionID string, authority SessionRecoveryAuthority) (SessionRecovery, error) {
	if c == nil || ctx == nil {
		return SessionRecovery{}, errors.New("agent loop: recovery authority is absent")
	}
	if err := authority.Task.ValidateIdentity(); err != nil {
		return SessionRecovery{}, err
	}
	if err := authority.Agent.ValidateIdentity(); err != nil {
		return SessionRecovery{}, err
	}
	if authority.Task.Agent != authority.Agent.ID || authority.Agent.ModelRecipe != c.identity.Recipe || authority.Model != c.identity.Model || authority.Catalog.Kind() != artifact.KindProfile {
		return SessionRecovery{}, errors.New("agent loop: recovery task, recipe, model, or catalog differs")
	}
	policies := slices.Clone(authority.Policies)
	sort.Slice(policies, func(i, j int) bool { return artifact.CompareID(policies[i], policies[j]) < 0 })
	if !slices.Equal(policies, authority.Agent.Policies) {
		return SessionRecovery{}, errors.New("agent loop: recovery policy authority differs")
	}
	active, found, err := c.store.ResolveAlias(ctx, "tool.catalog.active")
	if err != nil || !found || active != authority.Catalog {
		return SessionRecovery{}, errors.Join(errors.New("agent loop: recovery catalog is not active"), err)
	}
	if authority.Lease == nil {
		return SessionRecovery{}, errors.New("agent loop: recovery work lease is absent")
	}
	if err := worklease.ResolveOwner(ctx, c.store, *authority.Lease); err != nil {
		return SessionRecovery{}, err
	}
	for _, obligation := range authority.Obligations {
		if obligation.Task != authority.Task.ID || obligation.MutationEpoch > authority.MutationEpoch {
			return SessionRecovery{}, errors.New("agent loop: recovery obligation authority differs")
		}
	}
	contract, err := NewContractState(authority.Task, authority.Obligations)
	if err != nil {
		return SessionRecovery{}, err
	}
	for _, resolution := range authority.Resolutions {
		contract.AddResolution(resolution)
	}
	session, err := c.RestoreSession(ctx, sessionID)
	if err != nil {
		return SessionRecovery{}, err
	}
	session.Contract = contract
	next := fmt.Sprintf("%s-step-%d", session.ID, session.Steps+1)
	operation, err := MutationReceiptOperation(next)
	if err != nil {
		return SessionRecovery{}, err
	}
	receipt, receiptFound, err := runrecord.ResolveStageReceipt(ctx, c.store, operation, c.identity.Node)
	if err != nil {
		return SessionRecovery{}, err
	}
	result := SessionRecovery{Session: session, NextAction: next, Outstanding: contract.Outstanding()}
	if !receiptFound {
		return result, nil
	}
	if receipt.State != runrecord.StageRunning && receipt.State != runrecord.StageWaiting {
		return SessionRecovery{}, errors.New("agent loop: terminal mutation receipt lacks a committed interaction")
	}
	for _, checkpoint := range authority.Checkpoints {
		if err := checkpoint.ValidateIdentity(); err != nil {
			return SessionRecovery{}, err
		}
		if checkpoint.Operation == operation && checkpoint.MutationEpoch == authority.MutationEpoch {
			id := checkpoint.ID
			result.RestoreCheckpoint = &id
			result.NextAction = ""
			return result, nil
		}
	}
	return SessionRecovery{}, errors.New("agent loop: interrupted mutation lacks its exact checkpoint")
}
