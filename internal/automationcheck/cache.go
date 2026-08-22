package automationcheck

import (
	"context"
	"fmt"

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

// Compact removes entries written by the former input-keyed layout and any
// entry whose slot does not agree with its immutable invocation identity.
func (cache *EvidenceCache) Compact() {
	if cache == nil {
		return
	}
	for key, entry := range cache.Entries {
		if key != cacheKey(entry.Invocation) {
			delete(cache.Entries, key)
		}
	}
}

// RunCached executes or reuses one successful environment- and input-bound result.
func (cache *EvidenceCache) RunCached(ctx context.Context, invocation Invocation, input artifact.ID) (Evidence, bool, error) {
	if evidence, found := cache.Lookup(invocation, input); found {
		return evidence, true, nil
	}
	evidence, err := Run(ctx, invocation)
	if err == nil && !evidence.Skipped {
		cache.Record(invocation, input, evidence)
	}
	return evidence, false, err
}

// Lookup returns exact reusable evidence without executing the invocation.
func (cache *EvidenceCache) Lookup(invocation Invocation, input artifact.ID) (Evidence, bool) {
	entry, found := cache.Entries[cacheKey(invocation.ID)]
	if !found || entry.Input != input || entry.Outcome != runrecord.LanePassed {
		return Evidence{}, false
	}
	return Evidence{
		ID: entry.Evidence, InvocationID: entry.Invocation, Name: invocation.Check.Name,
		Phase: invocation.Check.Phase, Outcome: entry.Outcome, Reused: true,
	}, true
}

// Record replaces the stable invocation slot with one successful result.
func (cache *EvidenceCache) Record(invocation Invocation, input artifact.ID, evidence Evidence) {
	if evidence.Outcome == runrecord.LanePassed && !evidence.Skipped {
		cache.Entries[cacheKey(invocation.ID)] = CacheEntry{
			Invocation: invocation.ID, Input: input, Evidence: evidence.ID, Outcome: evidence.Outcome,
		}
	}
}

// PackageInvocation identifies one package/mode pair independently of its
// changing transitive inputs, so each pair owns exactly one replaceable slot.
func PackageInvocation(packagePath, mode string) (artifact.ID, error) {
	if packagePath == "" || mode == "" {
		return artifact.ID{}, fmt.Errorf("automation check: package invocation requires package and mode")
	}
	return artifact.JSONID(artifact.KindRecipe, struct {
		Package string `json:"package"`
		Mode    string `json:"mode"`
	}{packagePath, mode})
}

// PackageReusable reports whether an exact package input already passed.
func (cache *EvidenceCache) PackageReusable(packagePath, mode string, input artifact.ID) (bool, error) {
	invocation, err := PackageInvocation(packagePath, mode)
	if err != nil {
		return false, err
	}
	entry, found := cache.Entries[cacheKey(invocation)]
	return found && entry.Input == input && entry.Outcome == runrecord.LanePassed, nil
}

// RecordPackagePass replaces one package/mode slot with exact passing input.
func (cache *EvidenceCache) RecordPackagePass(packagePath, mode string, input artifact.ID) error {
	invocation, err := PackageInvocation(packagePath, mode)
	if err != nil {
		return err
	}
	evidence, err := artifact.JSONID(artifact.KindEvidence, struct {
		Invocation artifact.ID `json:"invocation"`
		Input      artifact.ID `json:"input"`
		Outcome    string      `json:"outcome"`
	}{invocation, input, string(runrecord.LanePassed)})
	if err != nil {
		return err
	}
	cache.Entries[cacheKey(invocation)] = CacheEntry{
		Invocation: invocation, Input: input, Evidence: evidence, Outcome: runrecord.LanePassed,
	}
	return nil
}

func cacheKey(invocation artifact.ID) string {
	return invocation.String()
}
