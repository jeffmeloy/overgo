package gate

import (
	"context"
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
		input, hasInput := inputs[check.ID]
		cacheCheck := phaseReusesEvidence(check.Check.Name) && hasInput
		// Checkpoint memo: stable slot + package-scoped input replace the
		// phase-wide fingerprint, so an unchanged checkpoint reuses across
		// candidates.
		slot := check
		if memoSlot, memoInput, memoised := g.memoSlot(check); memoised {
			slot, input, cacheCheck = memoSlot, memoInput, true
		}
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
