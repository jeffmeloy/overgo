package runrecord

import (
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// durableResult: one committed result artifact the log can point at.
func durableResult(t *testing.T, store *overgodb.Store, text string) artifact.ID {
	t.Helper()
	content, err := artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "text/plain", Schema: "overgo/test-result/v1"}.ContentBytes([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("durable-test/"+content.Descriptor.ID.String(), []artifact.Content{content}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return content.Descriptor.ID
}

// TestDurableAttemptInvocationContract pins: opening a unit advances its
// invocation count monotonically; a memo keyed by the caller is recorded once
// and read by every later invocation; a second, different completion is
// refused while an identical one is idempotent; waits and children are
// ordinal entries; a superseded attempt is refused at its first lookup;
// invalid kinds and keys are refused.
func TestDurableAttemptInvocationContract(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	unit := testutil.ArtifactID(t, artifact.KindEvidence, "durable unit")
	first, err := OpenDurableAttempt(t.Context(), store, unit)
	if err != nil || first.Unit != unit || first.Invocation != 1 {
		t.Fatalf("first attempt = %+v, %v", first, err)
	}
	if _, found, err := first.Lookup(t.Context(), store, DurableMemo, "fetch"); err != nil || found {
		t.Fatalf("unrecorded memo lookup = found=%t, %v", found, err)
	}
	resultA, resultB := durableResult(t, store, "a"), durableResult(t, store, "b")
	recorded, err := first.Record(t.Context(), store, DurableMemo, "fetch", resultA)
	if err != nil || recorded.Result != resultA || recorded.Invocation != 1 || recorded.Kind != DurableMemo {
		t.Fatalf("recorded memo = %+v, %v", recorded, err)
	}
	if again, err := first.Record(t.Context(), store, DurableMemo, "fetch", resultA); err != nil || again.ID != recorded.ID {
		t.Fatalf("identical completion was not idempotent: %+v, %v", again, err)
	}
	if held, err := first.Record(t.Context(), store, DurableMemo, "fetch", resultB); !errors.Is(err, ErrEntryRecorded) || held.Result != resultA {
		t.Fatalf("second completion = %+v, %v; want refusal holding the first result", held, err)
	}
	if _, err := first.Record(t.Context(), store, DurableChild, "1", resultB); err != nil {
		t.Fatal(err)
	}
	if _, err := first.Record(t.Context(), store, DurableWait, "1", resultA); err != nil {
		t.Fatal(err)
	}
	child, found, err := first.Lookup(t.Context(), store, DurableChild, "1")
	if err != nil || !found || child.Result != resultB {
		t.Fatalf("child entry = %+v found=%t, %v", child, found, err)
	}

	second, err := OpenDurableAttempt(t.Context(), store, unit)
	if err != nil || second.Invocation != 2 {
		t.Fatalf("second attempt = %+v, %v", second, err)
	}
	if _, _, err := first.Lookup(t.Context(), store, DurableMemo, "fetch"); !errors.Is(err, ErrStaleInvocation) {
		t.Fatalf("superseded attempt read the log: %v", err)
	}
	if _, err := first.Record(t.Context(), store, DurableMemo, "late", resultB); !errors.Is(err, ErrStaleInvocation) {
		t.Fatalf("superseded attempt recorded an entry: %v", err)
	}
	memo, found, err := second.Lookup(t.Context(), store, DurableMemo, "fetch")
	if err != nil || !found || memo.Result != resultA || memo.Invocation != 1 {
		t.Fatalf("later invocation memo = %+v found=%t, %v", memo, found, err)
	}
	if _, _, err := second.Lookup(t.Context(), store, DurableEntryKind("branch"), "x"); err == nil {
		t.Fatal("unknown entry kind was accepted")
	}
	if _, _, err := second.Lookup(t.Context(), store, DurableChild, "one"); err == nil {
		t.Fatal("a non-numeric child ordinal was accepted")
	}
	if _, _, err := second.Lookup(t.Context(), store, DurableMemo, ""); err == nil {
		t.Fatal("an empty memo key was accepted")
	}
	if err := (DurableAttempt{}).Current(t.Context(), store); err == nil {
		t.Fatal("an unbound attempt was current")
	}
}
