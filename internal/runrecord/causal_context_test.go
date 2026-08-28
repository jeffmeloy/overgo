package runrecord

import (
	"reflect"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestCausalContextLink pins the one shared context-to-projection adapter.
func TestCausalContextLink(t *testing.T) {
	id := func(label string) artifact.ID { return testutil.ArtifactID(t, artifact.KindEvidence, label) }
	rootID, subject := id("causal-link-root"), id("causal-link-subject")
	motivationA, motivationB := id("causal-link-motivation-a"), id("causal-link-motivation-b")
	root, err := NewCausalRoot(TriggerWebhook, rootID, motivationB, motivationA)
	if err != nil {
		t.Fatal(err)
	}
	context, err := root.Derive(TriggerRecovery, subject)
	if err != nil {
		t.Fatal(err)
	}
	execution := id("causal-link-execution")
	link, err := context.Link(execution)
	if err != nil {
		t.Fatal(err)
	}
	want := artifact.CausalLink{
		Execution: execution, Root: rootID, Trigger: string(TriggerRecovery),
		Subject: subject, Motivation: []artifact.ID{motivationA, motivationB},
	}
	if artifact.CompareID(motivationA, motivationB) > 0 {
		want.Motivation[0], want.Motivation[1] = want.Motivation[1], want.Motivation[0]
	}
	if !reflect.DeepEqual(link, want) {
		t.Fatalf("causal link = %+v, want %+v", link, want)
	}
	batch := artifact.Batch{}
	if err := BindCausality(&batch, execution, &context); err != nil {
		t.Fatal(err)
	}
	if len(batch.Causality) != 1 || !reflect.DeepEqual(batch.Causality[0], want) {
		t.Fatalf("bound causality = %+v, want %+v", batch.Causality, want)
	}
	context.Motivation[0] = id("mutated-motivation")
	if !reflect.DeepEqual(link, want) || !reflect.DeepEqual(batch.Causality[0], want) {
		t.Fatalf("causal link aliases its context: link=%+v batch=%+v", link, batch.Causality)
	}

	for name, refusal := range map[string]struct {
		execution artifact.ID
		context   CausalContext
	}{
		"absent execution":  {context: root},
		"foreign execution": {execution: testutil.ArtifactID(t, artifact.KindRun, "causal-link-run"), context: root},
		"execution is root": {execution: rootID, context: root},
		"invalid context":   {execution: execution, context: CausalContext{Trigger: "foreign", Root: rootID}},
	} {
		t.Run("refuses "+name, func(t *testing.T) {
			if _, err := refusal.context.Link(refusal.execution); err == nil {
				t.Fatal("accepted")
			}
		})
	}
}

// TestCausalContextIdentityAndValidation pins the causal contract:
// only originating triggers mint roots, every derivation copies the
// root identity verbatim through chains of any depth and shape (so
// delegation cycles cannot manufacture a new causal principal), retry
// ordinals count under the root, and field-trigger disagreement
// refuses.
func TestCausalContextIdentityAndValidation(t *testing.T) {
	id := func(label string) artifact.ID { return testutil.ArtifactID(t, artifact.KindEvidence, label) }
	rootEvidence := id("causal-webhook-event")
	motivationA, motivationB := id("motivating-regression"), id("motivating-review")
	if artifact.CompareID(motivationA, motivationB) > 0 {
		motivationA, motivationB = motivationB, motivationA
	}

	for _, trigger := range OriginatingTriggers {
		if _, err := NewCausalRoot(trigger, rootEvidence); err != nil {
			t.Fatalf("originating trigger %q refused a root: %v", trigger, err)
		}
	}
	for _, trigger := range DerivedTriggers {
		if _, err := NewCausalRoot(trigger, rootEvidence); err == nil {
			t.Fatalf("derived trigger %q minted a root", trigger)
		}
	}

	root, err := NewCausalRoot(TriggerWebhook, rootEvidence, motivationB, motivationA)
	if err != nil {
		t.Fatal(err)
	}
	if len(root.Motivation) != 2 || artifact.CompareID(root.Motivation[0], root.Motivation[1]) >= 0 {
		t.Fatalf("motivation not canonicalized: %+v", root.Motivation)
	}

	// A retry of a retry counts ordinals under one root.
	retry, err := root.Derive(TriggerRetry, id("attempt-0"))
	if err != nil {
		t.Fatal(err)
	}
	secondRetry, err := retry.Derive(TriggerRetry, id("attempt-1"))
	if err != nil {
		t.Fatal(err)
	}
	if retry.RetryOrdinal != 1 || secondRetry.RetryOrdinal != 2 {
		t.Fatalf("retry ordinals = %d then %d", retry.RetryOrdinal, secondRetry.RetryOrdinal)
	}

	// A delegation cycle -- A delegates to B, B back to A, then a
	// recovery and a rerun -- still answers to the original root.
	chain := root
	for _, hop := range []struct {
		trigger CausalTrigger
		subject artifact.ID
	}{
		{TriggerDelegation, id("execution-a")},
		{TriggerDelegation, id("execution-b")},
		{TriggerDelegation, id("execution-a")},
		{TriggerRecovery, id("lost-work")},
		{TriggerRerun, id("finished-run")},
	} {
		chain, err = chain.Derive(hop.trigger, hop.subject)
		if err != nil {
			t.Fatalf("hop %q: %v", hop.trigger, err)
		}
		if chain.Root != rootEvidence {
			t.Fatalf("hop %q re-minted the causal principal: %+v", hop.trigger, chain)
		}
	}
	if chain.ReplayOf != id("finished-run") || chain.DelegatedFrom.Valid() || chain.RecoveredFrom.Valid() {
		t.Fatalf("derivation carried stale subjects: %+v", chain)
	}
	if _, err := root.Derive(TriggerManual, id("execution-a")); err == nil {
		t.Fatal("originating trigger derived from a chain")
	}

	refusals := []struct {
		name    string
		context CausalContext
	}{
		{"foreign trigger", CausalContext{Trigger: "cosmic-ray", Root: rootEvidence}},
		{"root of a foreign kind", CausalContext{Trigger: TriggerManual,
			Root: testutil.ArtifactID(t, artifact.KindRun, "not-evidence")}},
		{"retry without its parent attempt", CausalContext{Trigger: TriggerRetry, Root: rootEvidence, RetryOrdinal: 1}},
		{"forged retry ordinal on a manual trigger", CausalContext{Trigger: TriggerManual, Root: rootEvidence, RetryOrdinal: 1}},
		{"retry with a zero ordinal", CausalContext{Trigger: TriggerRetry, Root: rootEvidence,
			ParentAttempt: id("attempt-0")}},
		{"delegation subject without its trigger", CausalContext{Trigger: TriggerManual, Root: rootEvidence,
			DelegatedFrom: id("execution-a")}},
		{"subject of a foreign kind", CausalContext{Trigger: TriggerRerun, Root: rootEvidence,
			ReplayOf: testutil.ArtifactID(t, artifact.KindRun, "run-not-evidence")}},
		{"unsorted motivation", CausalContext{Trigger: TriggerManual, Root: rootEvidence,
			Motivation: []artifact.ID{motivationB, motivationA}}},
		{"duplicate motivation", CausalContext{Trigger: TriggerManual, Root: rootEvidence,
			Motivation: []artifact.ID{motivationA, motivationA}}},
	}
	for _, refusal := range refusals {
		if err := refusal.context.Validate(); err == nil {
			t.Fatalf("%s: accepted", refusal.name)
		}
	}
}
