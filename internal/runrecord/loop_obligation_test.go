package runrecord

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestObligationCompletionReplay pins the durable obligation contract:
// obligations are typed records whose incompleteness is a store query,
// the replay completes exactly the obligations whose explicit
// predicates hold, completion survives a store reopen (the crash-lost
// case), and a second replay is idempotent.
func TestObligationCompletionReplay(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	subject := testutil.ArtifactID(t, artifact.KindEvidence, "obligation-subject")
	promised := testutil.ArtifactID(t, artifact.KindEvidence, "promised-evaluation")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "obligation/fixture/subject", Artifacts: []artifact.Descriptor{{ID: subject}}}); err != nil {
		t.Fatal(err)
	}
	now := int64(1_700_000_000_000_000_000)

	recovery, err := PublishLoopObligation(ctx, store, LoopObligation{
		Kind: ObligationRecoveryDispatch, Subject: subject,
		Predicate:     CompletionPredicate{Kind: PredicateAliasResolves, Alias: "obligation/fixture/recovered"},
		CreatedUnixNS: now,
	})
	if err != nil {
		t.Fatal(err)
	}
	evaluation, err := PublishLoopObligation(ctx, store, LoopObligation{
		Kind: ObligationReEvaluation, Subject: subject,
		Predicate:     CompletionPredicate{Kind: PredicateArtifactExists, Artifact: promised},
		CreatedUnixNS: now + 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	rollback, err := PublishLoopObligation(ctx, store, LoopObligation{
		Kind: ObligationRollbackNotification, Subject: subject,
		Predicate:     CompletionPredicate{Kind: PredicateAliasResolves, Alias: "obligation/fixture/notified"},
		CreatedUnixNS: now + 2,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Round one: the promised evaluation already exists, so exactly that
	// obligation completes; the two alias predicates stay due, oldest
	// first.
	if _, err := store.Commit(ctx, artifact.Batch{Key: "obligation/fixture/promised", Artifacts: []artifact.Descriptor{{ID: promised}}}); err != nil {
		t.Fatal(err)
	}
	replay, err := ReplayLoopObligations(ctx, store, now+10)
	if err != nil {
		t.Fatal(err)
	}
	if len(replay.Completed) != 1 || replay.Completed[0].Obligation != evaluation.ID ||
		replay.Completed[0].Evidence != promised {
		t.Fatalf("round one completions = %+v", replay.Completed)
	}
	if len(replay.Due) != 2 || replay.Due[0].ID != recovery.ID || replay.Due[1].ID != rollback.ID {
		t.Fatalf("round one due = %+v", replay.Due)
	}

	// The recovery dispatch lands, then the process dies: the next
	// replay runs against a reopened store with no remembered state and
	// repairs the follow-up from committed facts alone.
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "obligation/fixture/recovery-dispatch",
		Artifacts: []artifact.Descriptor{{ID: subject}},
		Aliases:   []artifact.AliasBinding{{Name: "obligation/fixture/recovered", Target: subject}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	replay, err = ReplayLoopObligations(ctx, store, now+20)
	if err != nil {
		t.Fatal(err)
	}
	if len(replay.Completed) != 1 || replay.Completed[0].Obligation != recovery.ID ||
		len(replay.Due) != 1 || replay.Due[0].ID != rollback.ID {
		t.Fatalf("post-crash replay = %+v", replay)
	}

	// Idempotence: completed obligations stay completed, the still-open
	// one stays due, and no duplicate completions appear.
	replay, err = ReplayLoopObligations(ctx, store, now+30)
	if err != nil {
		t.Fatal(err)
	}
	if len(replay.Completed) != 0 || len(replay.Due) != 1 || replay.Due[0].ID != rollback.ID {
		t.Fatalf("idempotent replay = %+v", replay)
	}

	refusals := []struct {
		name  string
		value LoopObligation
	}{
		{"foreign kind", LoopObligation{Kind: "someday", Subject: subject,
			Predicate: CompletionPredicate{Kind: PredicateAliasResolves, Alias: "x"}, CreatedUnixNS: now}},
		{"predicate naming both alias and artifact", LoopObligation{Kind: ObligationReEvaluation, Subject: subject,
			Predicate: CompletionPredicate{Kind: PredicateAliasResolves, Alias: "x", Artifact: promised}, CreatedUnixNS: now}},
		{"artifact predicate without an artifact", LoopObligation{Kind: ObligationReEvaluation, Subject: subject,
			Predicate: CompletionPredicate{Kind: PredicateArtifactExists}, CreatedUnixNS: now}},
		{"free-text predicate", LoopObligation{Kind: ObligationReEvaluation, Subject: subject,
			Predicate: CompletionPredicate{Kind: "operator-judgment", Alias: "x"}, CreatedUnixNS: now}},
	}
	for _, refusal := range refusals {
		if _, err := PublishLoopObligation(ctx, store, refusal.value); err == nil {
			t.Fatalf("%s: accepted", refusal.name)
		}
	}
}
