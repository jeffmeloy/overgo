package gate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

// executeChecks runs the planned DAG: reusable checks look up the cache by
// their phase input or checkpoint memo; each success is recorded and the
// cache written before the next check runs, so a killed process keeps every
// success it proved; drift, when set, guards each check against candidate
// tree changes; the persisted hook fires after each durable write.
func (g *gateContext) executeChecks(
	checks []automationcheck.Invocation,
	satisfied map[string]bool,
	inputs map[artifact.ID]artifact.ID,
	cache *automationcheck.EvidenceCache,
	drift func() error,
	deferred map[string]bool,
) ([]automationcheck.DAGResult, error) {
	g.modernPrior = nil
	for _, check := range checks {
		if check.Check.Name != modernCensusCheckName || cache == nil {
			continue
		}
		slot, _, _ := g.checkCacheKey(check, inputs[check.ID])
		entry, found := cache.Entries[slot.ID.String()]
		if found && entry.Source != nil && entry.Source.Definition == slot.ID {
			// Ask the cache to validate its original authority, not the new pair.
			slot.Authority = entry.Source.Authority
			if evidence, valid := cache.Lookup(slot, entry.Input); valid {
				g.modernPrior = evidence.Source
			}
		}
	}
	// The ledger registers every batch obligation, deferred ones included,
	// so an outstanding checkpoint stays recorded; only the executed set
	// enters the DAG, and a deferred check is never marked satisfied.
	ledger, err := g.openBatchEvidence(checks, inputs, cache)
	if err != nil {
		return nil, err
	}
	if g.terminal == nil {
		g.terminal = map[string]automationcheck.Evidence{}
	}
	if len(deferred) != 0 {
		checks = slices.DeleteFunc(slices.Clone(checks), func(check automationcheck.Invocation) bool { return deferred[check.Check.Name] })
	}
	var cacheMutex sync.Mutex
	return automationcheck.ExecuteDAG(context.Background(), checks, satisfied, func(ctx context.Context, check automationcheck.Invocation) (automationcheck.Evidence, error) {
		fmt.Fprintf(os.Stderr, gateProgressLine, check.Check.Name, runrecord.HeartbeatRunning)
		if drift != nil {
			if err := drift(); err != nil {
				return automationcheck.Evidence{}, err
			}
		}
		slot, input, cacheCheck := g.checkCacheKey(check, inputs[check.ID])
		if cacheCheck {
			cacheMutex.Lock()
			evidence, reused := cache.Lookup(slot, input)
			cacheMutex.Unlock()
			if reused {
				var err error
				reused, err = g.reusableOutput(check, evidence)
				if err != nil {
					return automationcheck.Evidence{}, err
				}
			}
			if reused {
				g.terminalMutex.Lock()
				g.terminal[check.Check.Name] = evidence
				g.terminalMutex.Unlock()
				return evidence, nil
			}
		}
		evidence, runErr := automationcheck.Run(ctx, check)
		if ledger != nil {
			if err := ledger.record(context.WithoutCancel(ctx), check.Check.Name, evidence); err != nil {
				return evidence, errors.Join(runErr, fmt.Errorf("persist batch obligation: %w", err))
			}
		}
		if runErr == nil && cacheCheck {
			cacheMutex.Lock()
			cache.Record(slot, input, evidence)
			// Durable before the next check: a kill past this point loses no
			// proven success; a write failure is audited, never silent.
			saveErr := g.saveRetryCache(*cache)
			if saveErr != nil {
				g.note("retry cache not saved after " + check.Check.Name + ": " + saveErr.Error())
			}
			cacheMutex.Unlock()
			if saveErr == nil && gateCheckPersistedHook != nil {
				gateCheckPersistedHook(check.Check.Name)
			}
		}
		if evidence.ID.Valid() {
			g.terminalMutex.Lock()
			g.terminal[check.Check.Name] = evidence
			g.terminalMutex.Unlock()
		}
		return evidence, runErr
	})
}

// checkCacheKey keeps execution and measurement eligibility identical. A
// checkpoint uses its package-scoped memo; ordinary phases require an input
// fingerprint and permission to reuse evidence.
func (g *gateContext) checkCacheKey(check automationcheck.Invocation, input artifact.ID) (automationcheck.Invocation, artifact.ID, bool) {
	if check.Check.Name == modernCensusCheckName && check.Authority != nil {
		check.ID = check.Authority.Definition
	}
	if slot, input, memoised := g.memoSlot(check); memoised {
		return slot, input, true
	}
	return check, input, input.Valid() && phaseReusesEvidence(check.Check.Name)
}

// Bind the actual memo input before execution; phase input alone is insufficient.
func (g *gateContext) bindCheckExecution(manifest automationcheck.ManifestPlan, check automationcheck.Invocation, input artifact.ID) (automationcheck.Invocation, error) {
	_, memoInput, _ := g.checkCacheKey(check, input)
	binding := &automationcheck.ReuseBinding{Input: memoInput, Environment: g.environment.ID}
	return automationcheck.BindManifestExecution(manifest, check, []artifact.ID{input}, binding)
}
