package gate

import (
	"context"
	"errors"
	"fmt"
	"os"
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
) ([]automationcheck.DAGResult, error) {
	ledger, err := g.openBatchEvidence(checks, inputs, cache)
	if err != nil {
		return nil, err
	}
	if ledger != nil {
		defer ledger.store.Close()
	}
	if g.terminal == nil {
		g.terminal = map[string]automationcheck.Evidence{}
	}
	var cacheMutex, terminalMutex sync.Mutex
	return automationcheck.ExecuteDAG(context.Background(), checks, satisfied, func(ctx context.Context, check automationcheck.Invocation) (automationcheck.Evidence, error) {
		fmt.Fprintf(os.Stderr, gateProgressLine, check.Check.Name, runrecord.HeartbeatRunning)
		if drift != nil {
			if err := drift(); err != nil {
				return automationcheck.Evidence{}, err
			}
		}
		slot, input, cacheCheck := g.checkCacheKey(check, inputs)
		if cacheCheck {
			cacheMutex.Lock()
			evidence, reused := cache.Lookup(slot, input)
			cacheMutex.Unlock()
			if reused {
				terminalMutex.Lock()
				g.terminal[check.Check.Name] = evidence
				terminalMutex.Unlock()
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
				g.audit = append(g.audit, "retry cache not saved after "+check.Check.Name+": "+saveErr.Error())
			}
			cacheMutex.Unlock()
			if saveErr == nil && gateCheckPersistedHook != nil {
				gateCheckPersistedHook(check.Check.Name)
			}
		}
		if evidence.ID.Valid() {
			terminalMutex.Lock()
			g.terminal[check.Check.Name] = evidence
			terminalMutex.Unlock()
		}
		return evidence, runErr
	})
}

// checkCacheKey keeps execution and measurement eligibility identical. A
// checkpoint uses its package-scoped memo; ordinary phases require an input
// fingerprint and permission to reuse evidence.
func (g *gateContext) checkCacheKey(check automationcheck.Invocation, inputs map[artifact.ID]artifact.ID) (automationcheck.Invocation, artifact.ID, bool) {
	if slot, input, memoised := g.memoSlot(check); memoised {
		return slot, input, true
	}
	input, found := inputs[check.ID]
	return check, input, found && phaseReusesEvidence(check.Check.Name)
}
