package overgodb

import (
	"context"
	"testing"

	"overgo/internal/artifact"
)

func TestVisitAliasEventsReturnsExactBoundedDeltas(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	content := func(value string) artifact.Content {
		t.Helper()
		payload := []byte(value)
		id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
		if err != nil {
			t.Fatal(err)
		}
		return artifact.Content{
			Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "text/plain"},
			Data:       payload,
		}
	}
	first, second := content("first"), content("second")
	alias := "fixture/active/policy"
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/alias-history/first", Contents: []artifact.Content{first},
		Aliases: []artifact.AliasBinding{{Name: alias, Target: first.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	secondCommit, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/alias-history/second", Contents: []artifact.Content{second},
		Aliases: []artifact.AliasBinding{{
			Name: alias, Target: second.Descriptor.ID, Previous: artifact.IDPointer(first.Descriptor.ID),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.sealActiveSegment(); err != nil {
		t.Fatal(err)
	}
	retireCommit, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/alias-history/retire",
		Aliases: []artifact.AliasBinding{{
			Name: alias, Target: second.Descriptor.ID, Previous: artifact.IDPointer(second.Descriptor.ID), Remove: true,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var events []AliasEvent
	err = store.VisitAliasEvents(context.Background(), AliasEventRange{
		Prefix: alias, FromSequence: 2, ToSequence: 3,
	}, func(event AliasEvent) error {
		events = append(events, event)
		return nil
	})
	if err != nil || len(events) != 2 {
		t.Fatalf("events = (%+v, %v)", events, err)
	}
	if events[0].Commit != (CommitView{Key: "fixture/alias-history/second", ID: secondCommit, Sequence: 2}) ||
		events[0].Binding.Name != alias || events[0].Binding.Target != second.Descriptor.ID ||
		events[0].Binding.Previous == nil || *events[0].Binding.Previous != first.Descriptor.ID || events[0].Binding.Remove {
		t.Fatalf("second event = %+v", events[0])
	}
	if events[1].Commit != (CommitView{Key: "fixture/alias-history/retire", ID: retireCommit, Sequence: 3}) ||
		events[1].Binding.Name != alias || events[1].Binding.Target != second.Descriptor.ID ||
		events[1].Binding.Previous == nil || *events[1].Binding.Previous != second.Descriptor.ID || !events[1].Binding.Remove {
		t.Fatalf("retirement event = %+v", events[1])
	}
	var all []AliasEvent
	if err := store.VisitAliasEvents(context.Background(), AliasEventRange{
		Prefix: alias, ToSequence: 3,
	}, func(event AliasEvent) error {
		all = append(all, event)
		return nil
	}); err != nil || len(all) != 3 {
		t.Fatalf("zero-start events = (%+v, %v)", all, err)
	}
	if err := store.VisitAliasEvents(context.Background(), AliasEventRange{
		Prefix: alias, FromSequence: 1, ToSequence: 4,
	}, func(AliasEvent) error { return nil }); err == nil {
		t.Fatal("range beyond the observed head was accepted")
	}
	if err := store.VisitAliasEvents(context.Background(), AliasEventRange{
		Prefix: alias,
	}, func(AliasEvent) error { return nil }); err == nil {
		t.Fatal("range without an observed head was accepted")
	}
}
