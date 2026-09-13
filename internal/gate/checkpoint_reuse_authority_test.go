package gate

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestCheckpointReuseCommitAuthority pins reuse provenance at commit
// admission and across restart. A checkpoint memo reused under a successor
// manifest plan keeps the original evidence identity in its source and is
// admitted under its own reuse identity; the restamped original, a reuse
// without its source and a self-citing source are refused. The proof
// survives the retry projection and the batch ledger: after a sibling
// failure and a restart the producer is reused with zero new execution.
func TestCheckpointReuseCommitAuthority(t *testing.T) {
	runner := func(context.Context, automationcheck.Invocation) (bool, string, error) { return false, "", nil }
	checks := []automationcheck.Check{
		{Descriptor: automationcheck.Descriptor{Name: "verify", Phase: runrecord.PhaseTest, Always: true}, Run: runner},
		{Descriptor: automationcheck.Descriptor{Name: "commit", Phase: runrecord.PhasePackage, Always: true, Dependencies: []string{"verify"}}, Run: runner},
	}
	invocations, err := automationcheck.Plan(checks, automationcheck.Impact{})
	if err != nil {
		t.Fatal(err)
	}
	input := testutil.ArtifactID(t, artifact.KindEvidence, "same checkpoint inputs")
	environment := lifecycleTestEnvironment(t)
	bind := func(tree string) (automationcheck.ManifestPlan, automationcheck.Invocation) {
		plan, err := automationcheck.BindManifestPlan(
			testutil.ArtifactID(t, artifact.KindProfile, "base"), testutil.ArtifactID(t, artifact.KindProfile, "candidate"),
			strings.Repeat("a", 64), strings.Repeat(tree, 64), automationcheck.Surface{Identity: "surface"}, automationcheck.Impact{}, invocations,
		)
		if err != nil {
			t.Fatal(err)
		}
		bound, err := automationcheck.BindManifestExecution(plan, invocations[0], []artifact.ID{input}, &automationcheck.ReuseBinding{Input: input, Environment: environment.ID})
		if err != nil {
			t.Fatal(err)
		}
		return plan, bound
	}
	firstPlan, first := bind("b")
	secondPlan, second := bind("c")
	g := gateContext{
		repo: t.TempDir(), environment: environment,
		checkpointMemos: map[string]checkpointMemoEntry{"verify": {slot: testutil.ArtifactID(t, artifact.KindRecipe, "stable checkpoint"), input: input}},
	}
	slot, _, _ := g.memoSlot(first)
	original, err := automationcheck.Run(t.Context(), first)
	if err != nil {
		t.Fatal(err)
	}
	cache := automationcheck.NewEvidenceCache(g.environment.ID)
	cache.Record(slot, input, original)
	if err := g.saveRetryCache(cache); err != nil {
		t.Fatal(err)
	}

	// Restart: the projection round-trips the proof.
	reloaded := g.loadRetryCache()
	nextSlot, _, _ := g.memoSlot(second)
	reused, found := reloaded.Lookup(nextSlot, input)
	if !found || reused.ID == original.ID || reused.Source == nil || reused.Source.Evidence != original.ID ||
		reused.Authority.Plan != secondPlan.ID || reused.Source.Authority.Plan != firstPlan.ID {
		t.Fatalf("reloaded reuse = %+v found=%t", reused, found)
	}
	if err := validateManifestCommitAdmission(secondPlan, map[string]automationcheck.Evidence{"verify": reused}); err != nil {
		t.Fatal(err)
	}
	restamped := original
	restamped.Authority = second.Authority
	restamped.Reused = true
	if validateManifestCommitAdmission(secondPlan, map[string]automationcheck.Evidence{"verify": restamped}) == nil {
		t.Fatal("restamped original admitted under the successor plan")
	}
	unproven := reused
	unproven.Source = nil
	if validateManifestCommitAdmission(secondPlan, map[string]automationcheck.Evidence{"verify": unproven}) == nil {
		t.Fatal("reuse without its original admitted")
	}
	selfCiting := reused
	selfCiting.Source = &automationcheck.ReuseSource{}
	*selfCiting.Source = *reused.Source
	selfCiting.Source.Evidence = reused.ID
	if selfCiting.ID, err = selfCiting.Identity(); err != nil {
		t.Fatal(err)
	}
	if validateManifestCommitAdmission(secondPlan, map[string]automationcheck.Evidence{"verify": selfCiting}) == nil {
		t.Fatal("self-citing reuse admitted")
	}

	// A projection entry from the former layout proves nothing: the
	// checkpoint executes once more.
	legacy := automationcheck.NewEvidenceCache(g.environment.ID)
	legacy.Entries[slot.ID.String()] = automationcheck.CacheEntry{Invocation: slot.ID, Input: input, Evidence: original.ID, Outcome: runrecord.LanePassed}
	if err := g.saveRetryCache(legacy); err != nil {
		t.Fatal(err)
	}
	stale := g.loadRetryCache()
	if _, found := stale.Lookup(nextSlot, input); found {
		t.Fatal("unproven projection entry reused")
	}

	// Sibling failure, then restart from the batch ledger with the retry
	// projection removed: the producer is reused with zero new execution and
	// the consumer fails again.
	producerRun, batch, tree := verificationBatchFixture(t, "failure")
	producerRun.environment = lifecycleTestEnvironment(t)
	bound := persistenceInvocations(t, producerRun, batch, tree)
	attempt := producerRun.loadRetryCache()
	results, err := producerRun.executeChecks(bound, nil, map[artifact.ID]artifact.ID{}, &attempt, nil)
	if err != nil {
		t.Fatal(err)
	}
	var producerEvidence automationcheck.Evidence
	for _, result := range results {
		switch result.Invocation.Check.Name {
		case "acceptance-producer":
			if result.Err != nil || result.Evidence.Reused || !result.Evidence.ID.Valid() {
				t.Fatalf("producer did not execute: %+v", result)
			}
			producerEvidence = result.Evidence
		case "acceptance-consumer":
			if result.Err == nil {
				t.Fatal("failing consumer passed")
			}
		}
	}
	if err := os.Remove(filepath.Join(producerRun.repo, filepath.FromSlash(gateRetryFile))); err != nil {
		t.Fatal(err)
	}
	restart, batch, tree := verificationBatchContext(t, producerRun.repo)
	restart.environment = producerRun.environment
	bound = persistenceInvocations(t, restart, batch, tree)
	attempt = restart.loadRetryCache()
	results, err = restart.executeChecks(bound, nil, map[artifact.ID]artifact.ID{}, &attempt, nil)
	if err != nil {
		t.Fatal(err)
	}
	var producerDefinition artifact.ID
	for _, invocation := range bound {
		if invocation.Check.Name == "acceptance-producer" {
			producerDefinition = invocation.Authority.Definition
		}
	}
	reusedProducer := false
	for _, result := range results {
		switch result.Invocation.Check.Name {
		case "acceptance-producer":
			evidence := result.Evidence
			if result.Err != nil || !evidence.Reused || evidence.Source == nil || evidence.Source.Evidence != producerEvidence.ID ||
				evidence.Authority.Plan != restart.manifestPlan.ID {
				t.Fatalf("producer was not reused from the ledger: %+v", result)
			}
			if err := automationcheck.ValidateReuseAuthority(evidence, producerDefinition); err != nil {
				t.Fatal(err)
			}
			reusedProducer = true
		case "acceptance-consumer":
			if result.Err == nil || result.Evidence.Reused {
				t.Fatalf("failing sibling was reused or passed: %+v", result)
			}
		}
	}
	if !reusedProducer {
		t.Fatal("producer result absent after restart")
	}
	if _, found := attempt.Lookup(bound[slotIndex(bound, "acceptance-consumer")], input); found {
		t.Fatal("failed consumer entered the cache")
	}
}

func slotIndex(invocations []automationcheck.Invocation, name string) int {
	return slices.IndexFunc(invocations, func(invocation automationcheck.Invocation) bool { return invocation.Check.Name == name })
}
