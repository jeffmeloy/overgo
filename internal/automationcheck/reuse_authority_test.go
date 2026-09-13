package automationcheck

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestCheckpointReuseAuthority pins reuse provenance at the cache owner. A
// reused result is a new evidence identity under the successor's authority
// that cites the original execution; the original's identity never changes.
// An entry without proof, in another environment, for another definition,
// input or original is not reused; chained reuse cites the first execution
// without nesting; identities verify and tampering is refused.
func TestCheckpointReuseAuthority(t *testing.T) {
	runs := 0
	runner := func(context.Context, Invocation) (bool, string, error) { return false, "", nil }
	checks := []Check{
		{Descriptor: Descriptor{Name: "verify", Phase: runrecord.PhaseTest, Always: true},
			Run: func(context.Context, Invocation) (bool, string, error) {
				runs++
				return false, "verified checkpoint", nil
			}},
		{Descriptor: Descriptor{Name: "commit", Phase: runrecord.PhasePackage, Always: true, Dependencies: []string{"verify"}}, Run: runner},
	}
	invocations, err := Plan(checks, Impact{})
	if err != nil {
		t.Fatal(err)
	}
	definition := invocations[0].ID
	input := testutil.ArtifactID(t, artifact.KindEvidence, "same checkpoint inputs")
	otherInput := testutil.ArtifactID(t, artifact.KindEvidence, "other checkpoint inputs")
	environment := testutil.ArtifactID(t, artifact.KindProfile, "environment")
	// The gate's checkpoint memo keeps one slot across candidates; the
	// bound invocation changes with every plan.
	slotID := testutil.ArtifactID(t, artifact.KindRecipe, "stable checkpoint slot")
	bind := func(tree string, binding ReuseBinding) (ManifestPlan, Invocation, Invocation) {
		plan, err := BindManifestPlan(
			testutil.ArtifactID(t, artifact.KindProfile, "base"), testutil.ArtifactID(t, artifact.KindProfile, "candidate"),
			strings.Repeat("a", 64), strings.Repeat(tree, 64), Surface{Identity: "surface"}, Impact{}, invocations,
		)
		if err != nil {
			t.Fatal(err)
		}
		bound, err := BindManifestExecution(plan, invocations[0], []artifact.ID{binding.Input}, &binding)
		if err != nil {
			t.Fatal(err)
		}
		slot := bound
		slot.ID = slotID
		return plan, bound, slot
	}
	binding := ReuseBinding{Input: input, Environment: environment}
	planA, boundA, slotA := bind("b", binding)
	planB, _, slotB := bind("c", binding)
	_, _, slotC := bind("d", binding)
	cache := NewEvidenceCache(environment)

	original, err := Run(t.Context(), boundA)
	if err != nil || runs != 1 || original.VerifyIdentity() != nil {
		t.Fatalf("original = %+v, %v, runs=%d", original, err, runs)
	}
	cache.Record(slotA, input, original)
	key := cacheKey(slotID)
	if !cache.Entries[key].Proven() {
		t.Fatalf("recorded entry is unproven: %+v", cache.Entries[key])
	}

	reused, found := cache.Lookup(slotB, input)
	if !found || runs != 1 || !reused.Reused || reused.ID == original.ID || reused.InvocationID != slotID ||
		reused.Authority == nil || reused.Authority.Plan != planB.ID || reused.Authority.Definition != definition ||
		reused.Source == nil || reused.Source.Evidence != original.ID || reused.Source.InvocationID != boundA.ID ||
		reused.Source.Authority == nil || reused.Source.Authority.Plan != planA.ID || reused.Source.Definition != definition ||
		reused.Source.Environment != environment || reused.Source.Input != input {
		t.Fatalf("successor reuse = %+v found=%t runs=%d", reused, found, runs)
	}
	if err := reused.VerifyIdentity(); err != nil {
		t.Fatal(err)
	}
	if err := ValidateReuseAuthority(reused, definition); err != nil {
		t.Fatal(err)
	}
	lineage := EvidenceLineage(reused)
	for _, want := range []artifact.ID{planA.ID, planB.ID, original.ID, definition, input, environment} {
		if !slices.Contains(lineage, want) {
			t.Fatalf("reuse lineage %v omits %s", lineage, want)
		}
	}
	retry, found := cache.Lookup(slotA, input)
	if !found || retry.ID == original.ID || retry.Authority.Plan != planA.ID || retry.Source.Evidence != original.ID {
		t.Fatalf("same-plan retry = %+v found=%t", retry, found)
	}

	// The former defect: the original's identity restamped with the
	// successor's authority.
	restamped := original
	restamped.Authority = cloneExecutionAuthority(slotB.Authority)
	restamped.Reused = true
	if restamped.VerifyIdentity() == nil || ValidateReuseAuthority(restamped, definition) == nil {
		t.Fatal("restamped original kept a verifiable identity")
	}
	if ValidateReuseAuthority(original, definition) == nil {
		t.Fatal("an executed result passed as reuse")
	}
	tamper := func(name string, mutate func(evidence *Evidence)) {
		t.Helper()
		tampered := reused
		tampered.Authority = cloneExecutionAuthority(reused.Authority)
		tampered.Source = cloneReuseSource(reused.Source)
		mutate(&tampered)
		if ValidateReuseAuthority(tampered, definition) == nil {
			t.Fatalf("%s admitted", name)
		}
	}
	tamper("authority plan", func(evidence *Evidence) { evidence.Authority.Plan = planA.ID })
	tamper("missing source", func(evidence *Evidence) { evidence.Source = nil })
	reidentify := func(evidence *Evidence) {
		id, err := evidence.Identity()
		if err != nil {
			t.Fatal(err)
		}
		evidence.ID = id
	}
	tamper("self-citing source", func(evidence *Evidence) { evidence.Source.Evidence = reused.ID; reidentify(evidence) })
	tamper("foreign source definition", func(evidence *Evidence) {
		evidence.Source.Definition = invocations[1].ID
		evidence.Source.Authority.Definition = invocations[1].ID
		reidentify(evidence)
	})
	tamper("unbound source", func(evidence *Evidence) { evidence.Source.Authority = nil; reidentify(evidence) })
	tamper("changed original detail", func(evidence *Evidence) { evidence.Source.Detail = "another result"; reidentify(evidence) })
	tamper("failed reuse", func(evidence *Evidence) { evidence.Outcome = runrecord.LaneFailed; reidentify(evidence) })
	tamper("inapplicable reuse", func(evidence *Evidence) { evidence.Inapplicable = true; reidentify(evidence) })
	tamper("foreign reuse definition", func(evidence *Evidence) { evidence.Authority.Definition = invocations[1].ID; reidentify(evidence) })
	tamper("successor input", func(evidence *Evidence) { evidence.Authority.Reuse.Input = otherInput; reidentify(evidence) })
	tamper("successor environment", func(evidence *Evidence) { evidence.Authority.Reuse.Environment = otherInput; reidentify(evidence) })

	proven := cache.Entries[key]
	for _, change := range []struct {
		name    string
		binding ReuseBinding
	}{
		{"relabel input", ReuseBinding{Input: otherInput, Environment: environment}},
		{"relabel environment", ReuseBinding{Input: input, Environment: otherInput}},
		{"swap input and environment", ReuseBinding{Input: environment, Environment: input}},
	} {
		_, successor, lookup := bind("e", change.binding)
		entry := proven
		entry.Source = cloneReuseSource(proven.Source)
		entry.Input, entry.Source.Input = change.binding.Input, change.binding.Input
		entry.Source.Environment = change.binding.Environment
		changedCache := NewEvidenceCache(change.binding.Environment)
		changedCache.Entries[key] = entry
		if _, found := changedCache.Lookup(lookup, change.binding.Input); found {
			t.Fatalf("%s reused an execution bound to other inputs", change.name)
		}
		forged := reused
		forged.Authority, forged.Source = successor.Authority, entry.Source
		reidentify(&forged)
		if ValidateReuseAuthority(forged, definition) == nil {
			t.Fatalf("%s admitted a reidentified counterfeit reuse", change.name)
		}
	}
	legacyOriginal := original
	legacyOriginal.Authority = cloneExecutionAuthority(original.Authority)
	legacyOriginal.Authority.Reuse = nil
	reidentify(&legacyOriginal)
	if _, ok := NewCacheEntry(slotID, input, environment, legacyOriginal); ok {
		t.Fatal("legacy execution without immutable reuse inputs was recorded")
	}
	refuse := func(name string, entry CacheEntry, lookupCache *EvidenceCache, lookupInput artifact.ID) {
		t.Helper()
		lookupCache.Entries[key] = entry
		if _, found := lookupCache.Lookup(slotB, lookupInput); found {
			t.Fatalf("%s reused", name)
		}
		lookupCache.Entries[key] = proven
	}
	refuse("other input", proven, &cache, otherInput)
	other := NewEvidenceCache(testutil.ArtifactID(t, artifact.KindProfile, "other environment"))
	refuse("foreign environment", proven, &other, input)
	entry := proven
	entry.Source = cloneReuseSource(proven.Source)
	entry.Source.Definition = invocations[1].ID
	refuse("other definition", entry, &cache, input)
	entry.Source = cloneReuseSource(proven.Source)
	entry.Source.Evidence = otherInput
	refuse("other original", entry, &cache, input)
	entry.Source = cloneReuseSource(proven.Source)
	entry.Source.Input = otherInput
	refuse("other source input", entry, &cache, input)
	legacy := CacheEntry{Invocation: slotID, Input: input, Evidence: original.ID, Outcome: runrecord.LanePassed}
	refuse("unproven legacy entry", legacy, &cache, input)

	// An unproven entry executes once more, then reuse resumes with proof.
	cache.Entries[key] = legacy
	rerun, wasReused, err := cache.RunCached(t.Context(), slotB, input)
	if err != nil || wasReused || runs != 2 || rerun.Reused || !cache.Entries[key].Proven() {
		t.Fatalf("legacy rerun = %+v reused=%t err=%v runs=%d", rerun, wasReused, err, runs)
	}
	if _, found := cache.Lookup(slotB, input); !found {
		t.Fatal("proven rerun was not reused")
	}
	cache.Entries[key] = proven

	// Chained reuse records the first execution; the entry does not grow.
	cache.Record(slotB, input, reused)
	if !reflect.DeepEqual(cache.Entries[key], proven) {
		t.Fatalf("chained record changed the entry: %+v", cache.Entries[key])
	}
	chained, found := cache.Lookup(slotC, input)
	if !found || chained.Source.Evidence != original.ID || chained.Source.Authority.Plan != planA.ID {
		t.Fatalf("chained reuse = %+v found=%t", chained, found)
	}

	if _, ok := NewCacheEntry(slotID, input, environment, Evidence{ID: original.ID, Outcome: runrecord.LanePassed, Inapplicable: true}); ok {
		t.Fatal("inapplicable result recorded")
	}
	if _, ok := NewCacheEntry(slotID, input, environment, Evidence{Outcome: runrecord.LanePassed}); ok {
		t.Fatal("unidentified result recorded")
	}
	unproven := reused
	unproven.Source = nil
	if _, ok := NewCacheEntry(slotID, input, environment, unproven); ok {
		t.Fatal("reuse without its original recorded")
	}
	concealedReuse := reused
	concealedReuse.Reused = false
	reidentify(&concealedReuse)
	if _, ok := NewCacheEntry(slotID, input, environment, concealedReuse); ok {
		t.Fatal("reuse source recorded as an original execution")
	}
	for _, candidate := range []struct {
		name               string
		input, environment artifact.ID
		evidence           Evidence
	}{
		{"changed input", otherInput, environment, reused},
		{"changed environment", input, other.Environment, reused},
		{"restamped original", input, environment, restamped},
	} {
		if _, ok := NewCacheEntry(slotID, candidate.input, candidate.environment, candidate.evidence); ok {
			t.Fatalf("record relabeled %s", candidate.name)
		}
	}
}
