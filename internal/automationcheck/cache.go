package automationcheck

import (
	"context"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// CacheEntry binds one successful result to its exact invocation and inputs.
type CacheEntry struct {
	Invocation artifact.ID           `json:"invocation"`
	Input      artifact.ID           `json:"input"`
	Evidence   artifact.ID           `json:"evidence"`
	Outcome    runrecord.LaneOutcome `json:"outcome"`
}

// EvidenceCache contains environment-bound reusable check evidence.
type EvidenceCache struct {
	Environment artifact.ID           `json:"environment"`
	Entries     map[string]CacheEntry `json:"entries"`
}

// NewEvidenceCache creates an empty cache for one exact environment.
func NewEvidenceCache(environment artifact.ID) EvidenceCache {
	return EvidenceCache{Environment: environment, Entries: map[string]CacheEntry{}}
}

// Reusable reports whether cache evidence belongs to the current environment.
func (cache EvidenceCache) Reusable(environment artifact.ID) bool {
	return environment.Valid() && cache.Environment == environment && cache.Entries != nil
}

// RunCached executes or reuses one successful environment- and input-bound result.
func (cache *EvidenceCache) RunCached(ctx context.Context, invocation Invocation, input artifact.ID) (Evidence, bool, error) {
	key := cacheKey(invocation.ID, input)
	if entry, found := cache.Entries[key]; found && entry.Outcome == runrecord.LanePassed {
		return Evidence{
			ID: entry.Evidence, InvocationID: entry.Invocation, Name: invocation.Check.Name,
			Phase: invocation.Check.Phase, Outcome: entry.Outcome, Skipped: true,
		}, true, nil
	}
	evidence, err := Run(ctx, invocation)
	if err == nil && !evidence.Skipped {
		cache.Entries[key] = CacheEntry{Invocation: invocation.ID, Input: input, Evidence: evidence.ID, Outcome: evidence.Outcome}
	}
	return evidence, false, err
}

func cacheKey(invocation, input artifact.ID) string {
	return invocation.String() + "\x00" + input.String()
}
