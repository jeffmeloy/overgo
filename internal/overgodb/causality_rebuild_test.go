package overgodb

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestCausalityProjectionRebuild proves operational causality is a bounded,
// rebuildable projection over canonical journal facts. Root, descendant,
// retry, replay, and evidence-jump queries survive checkpoint loading and a
// fresh-store rebuild, while missing roots and cycles refuse publication.
func TestCausalityProjectionRebuild(t *testing.T) {
	ctx := t.Context()
	id := func(label string) artifact.ID { return testutil.ArtifactID(t, artifact.KindEvidence, label) }
	root, motivation := id("causal-root"), id("causal-motivation")
	attempt, retry, replay, evaluation := id("attempt"), id("retry"), id("replay"), id("evaluation")
	all := []artifact.ID{root, motivation, attempt, retry, replay, evaluation}
	descriptors := make([]artifact.Descriptor, len(all))
	for index, value := range all {
		descriptors[index] = artifact.Descriptor{ID: value}
	}
	links := []artifact.CausalLink{
		{Execution: attempt, Root: root, Trigger: "controller-proposal", Motivation: []artifact.ID{motivation}},
		{Execution: retry, Root: root, Trigger: "retry", Subject: attempt, Motivation: []artifact.ID{motivation}},
		{Execution: replay, Root: root, Trigger: "rerun", Subject: attempt, Motivation: []artifact.ID{motivation}},
		{Execution: evaluation, Root: root, Trigger: "delegation", Subject: retry, Motivation: []artifact.ID{motivation}},
	}

	storeRoot := t.TempDir()
	store, err := Open(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "causality/corpus", Artifacts: descriptors, Causality: links,
		Aliases: []artifact.AliasBinding{{Name: "causality/live", Target: evaluation}},
	}); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	assertCausalityProjection(t, store, head, sequence, root, motivation, attempt, retry, replay, evaluation)
	if _, err := store.Snapshot(ctx); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	checkpointed, err := OpenReadOnly(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	if !checkpointed.SnapshotReplay().Loaded {
		t.Fatal("causality projection did not load from its anchored checkpoint set")
	}
	assertCausalityProjection(t, checkpointed, head, sequence, root, motivation, attempt, retry, replay, evaluation)

	rebuiltRoot := t.TempDir()
	rebuildReport, err := Rebuild(ctx, checkpointed, rebuiltRoot, nil)
	if err != nil || rebuildReport.CausalLinks != len(links) {
		t.Fatalf("rebuild report = (%+v, %v)", rebuildReport, err)
	}
	compactedRoot := t.TempDir()
	compactionReport, err := Compact(ctx, checkpointed, compactedRoot)
	if err != nil || compactionReport.Causality != 3 {
		t.Fatalf("compaction report = (%+v, %v)", compactionReport, err)
	}
	if err := checkpointed.Close(); err != nil {
		t.Fatal(err)
	}
	rebuilt, err := OpenReadOnly(rebuiltRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer rebuilt.Close()
	rebuiltHead, rebuiltSequence := rebuilt.Head()
	assertCausalityProjection(t, rebuilt, rebuiltHead, rebuiltSequence, root, motivation, attempt, retry, replay, evaluation)
	compacted, err := OpenReadOnly(compactedRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer compacted.Close()
	compactedResult, err := compacted.QueryCausality(ctx, CausalityQuery{Root: &root, MaxResults: 8})
	if err != nil || compactedResult.Matched != 3 || !containsCausalExecutions(compactedResult.Links, attempt, retry, evaluation) {
		t.Fatalf("compacted causality = (%+v, %v)", compactedResult, err)
	}

	refusal, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer refusal.Close()
	if _, err := refusal.Commit(ctx, artifact.Batch{
		Key: "causality/missing-root", Artifacts: []artifact.Descriptor{{ID: attempt}},
		Causality: []artifact.CausalLink{{Execution: attempt, Root: root, Trigger: "manual"}},
	}); err == nil {
		t.Fatal("causality projection accepted a missing root")
	}
	if _, err := refusal.Commit(ctx, artifact.Batch{Key: "causality/cycle-base", Artifacts: []artifact.Descriptor{
		{ID: root}, {ID: attempt}, {ID: retry},
	}, Causality: []artifact.CausalLink{{Execution: attempt, Root: root, Trigger: "retry", Subject: retry}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := refusal.Commit(ctx, artifact.Batch{Key: "causality/cycle-close", Causality: []artifact.CausalLink{
		{Execution: retry, Root: root, Trigger: "retry", Subject: attempt},
	}}); err == nil {
		t.Fatal("causality projection accepted a cycle")
	}
}

func assertCausalityProjection(
	t *testing.T,
	store *Store,
	head artifact.CommitID,
	sequence uint64,
	root, motivation, attempt, retry, replay, evaluation artifact.ID,
) {
	t.Helper()
	rootResult, err := store.QueryCausality(t.Context(), CausalityQuery{Root: &root, MaxResults: 8})
	if err != nil || rootResult.Head != head || rootResult.Sequence != sequence ||
		rootResult.ProjectionVersion != CausalityProjectionVersion || rootResult.Matched != 4 || rootResult.Truncated {
		t.Fatalf("root result = (%+v, %v)", rootResult, err)
	}
	bounded, err := store.QueryCausality(t.Context(), CausalityQuery{Root: &root, MaxResults: 2})
	if err != nil || bounded.Matched != 4 || len(bounded.Links) != 2 || !bounded.Truncated {
		t.Fatalf("bounded root result = (%+v, %v)", bounded, err)
	}
	descendants, err := store.QueryCausality(t.Context(), CausalityQuery{
		DescendantOf: &attempt, MaxDepth: 2, MaxResults: 8,
	})
	if err != nil || descendants.Matched != 3 || !containsCausalExecutions(descendants.Links, retry, replay, evaluation) {
		t.Fatalf("descendants = (%+v, %v)", descendants, err)
	}
	retries, err := store.QueryCausality(t.Context(), CausalityQuery{Root: &root, Trigger: "retry", MaxResults: 8})
	if err != nil || retries.Matched != 1 || retries.Links[0].Execution != retry {
		t.Fatalf("retries = (%+v, %v)", retries, err)
	}
	replays, err := store.QueryCausality(t.Context(), CausalityQuery{Trigger: "rerun", MaxResults: 8})
	if err != nil || replays.Matched != 1 || replays.Links[0].Execution != replay {
		t.Fatalf("replays = (%+v, %v)", replays, err)
	}
	jump, err := store.QueryCausality(t.Context(), CausalityQuery{Evidence: &motivation, MaxResults: 8})
	if err != nil || jump.Matched != 4 {
		t.Fatalf("evidence jump = (%+v, %v)", jump, err)
	}
}

func containsCausalExecutions(links []artifact.CausalLink, required ...artifact.ID) bool {
	for _, id := range required {
		if !slices.ContainsFunc(links, func(link artifact.CausalLink) bool { return link.Execution == id }) {
			return false
		}
	}
	return true
}
