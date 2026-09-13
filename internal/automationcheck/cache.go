package automationcheck

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// CacheEntry binds one successful result to its exact invocation and inputs.
// Source is the original execution the entry stands on; an entry without it
// proves nothing and executes once more before it can be reused.
type CacheEntry struct {
	Invocation artifact.ID           `json:"invocation"`
	Input      artifact.ID           `json:"input"`
	Evidence   artifact.ID           `json:"evidence"`
	Outcome    runrecord.LaneOutcome `json:"outcome"`
	Source     *ReuseSource          `json:"source,omitempty"`
}

// NewCacheEntry records one passed result under a slot, input and
// environment. A reused result is recorded through its original, so chained
// reuse never nests sources; a failed, inapplicable, unidentified or
// unproven result yields no entry.
func NewCacheEntry(slot, input, environment artifact.ID, evidence Evidence) (CacheEntry, bool) {
	if !slot.Valid() || !input.Valid() || evidence.Outcome != runrecord.LanePassed ||
		evidence.Inapplicable || evidence.VerifyIdentity() != nil {
		return CacheEntry{}, false
	}
	source := ReuseSource{
		Evidence: evidence.ID, InvocationID: evidence.InvocationID, Authority: cloneExecutionAuthority(evidence.Authority),
		Definition: executionDefinition(evidence.Authority, evidence.InvocationID), Environment: environment, Input: input, Detail: evidence.Detail,
	}
	if evidence.Reused {
		if !evidence.Source.complete() || evidence.Source.Input != input ||
			evidence.Source.Environment != environment || evidence.Source.Definition != source.Definition {
			return CacheEntry{}, false
		}
		source = *cloneReuseSource(evidence.Source)
	}
	if !source.complete() {
		return CacheEntry{}, false
	}
	return CacheEntry{Invocation: slot, Input: input, Evidence: source.Evidence, Outcome: evidence.Outcome, Source: &source}, true
}

// Proven reports whether the entry carries a complete original execution
// that agrees with its own evidence and input.
func (entry CacheEntry) Proven() bool {
	return entry.Outcome == runrecord.LanePassed && entry.Source.complete() &&
		entry.Source.Evidence == entry.Evidence && entry.Source.Input == entry.Input
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
	if err == nil && !evidence.Inapplicable {
		cache.Record(invocation, input, evidence)
	}
	return evidence, false, err
}

// Lookup returns a reused result without executing the invocation: a new
// evidence identity under the invocation's own authority that cites the
// original execution. The original must be proven, recorded in this
// environment under the same input, and verify the same definition.
func (cache *EvidenceCache) Lookup(invocation Invocation, input artifact.ID) (Evidence, bool) {
	entry, found := cache.Entries[cacheKey(invocation.ID)]
	if !found || entry.Input != input || !entry.Proven() ||
		entry.Source.Environment != cache.Environment || entry.Source.Definition != executionDefinition(invocation.Authority, invocation.ID) {
		return Evidence{}, false
	}
	if invocation.Authority != nil && !(ReuseBinding{Input: input, Environment: cache.Environment}).matches(invocation.Authority) {
		return Evidence{}, false
	}
	evidence := Evidence{
		InvocationID: invocation.ID, Authority: cloneExecutionAuthority(invocation.Authority), Name: invocation.Check.Name,
		Phase: invocation.Check.Phase, Outcome: entry.Outcome, Reused: true, Source: cloneReuseSource(entry.Source),
	}
	id, err := evidence.Identity()
	if err != nil {
		return Evidence{}, false
	}
	evidence.ID = id
	return evidence, true
}

// Record replaces the stable invocation slot with one successful result.
func (cache *EvidenceCache) Record(invocation Invocation, input artifact.ID, evidence Evidence) {
	if entry, ok := NewCacheEntry(invocation.ID, input, cache.Environment, evidence); ok {
		cache.Entries[cacheKey(invocation.ID)] = entry
	}
}

// PackageInvocation identifies one package/mode pair independently of its
// changing transitive inputs, so each pair owns exactly one replaceable slot.
// Its policy excludes historical group-level passes lacking complete package
// evidence; those entries must be reverified under per-package admission.
func PackageInvocation(packagePath, mode string) (artifact.ID, error) {
	if packagePath == "" || mode == "" {
		return artifact.ID{}, fmt.Errorf("automation check: package invocation requires package and mode")
	}
	policy := "go-test-complete-package/v1"
	if mode == "short" {
		policy = "go-test-short-profile/v1"
	}
	return artifact.JSONID(artifact.KindRecipe, struct {
		Policy  string `json:"policy"`
		Package string `json:"package"`
		Mode    string `json:"mode"`
	}{policy, packagePath, mode})
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
