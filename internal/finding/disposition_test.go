package finding

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestDispositionSupersedesAnOpenFinding(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	original, batch, err := NewTextBatch("routing sizes are manifest sizes", SeverityHigh,
		[]string{"internal/evaluation"}, []string{"cheapest chose the 12B"}, "price by component bytes", "grep size")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if _, disposed, err := Disposition(ctx, store, original.ID); err != nil || disposed {
		t.Fatalf("fresh finding disposed = %v, %v", disposed, err)
	}
	closed, closeBatch, err := NewDispositionBatch(ctx, store, original, StatusClosed, "fixed in dc8a485c")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, closeBatch); err != nil {
		t.Fatal(err)
	}
	if closed.Status != StatusClosed || closed.Title != original.Title || closed.Severity != original.Severity ||
		len(closed.Evidence) != 2 || !slices.Contains(closed.Evidence, original.ID) {
		t.Fatalf("disposition = %+v", closed)
	}
	if disposition, disposed, err := Disposition(ctx, store, original.ID); err != nil || !disposed || disposition.ID != closed.ID {
		t.Fatalf("disposed finding reads live: %v, %v", disposed, err)
	}
	target, found, err := artifact.ResolveAlias(ctx, store, DispositionAliasPrefix+original.ID.String())
	if err != nil || !found || target != closed.ID {
		t.Fatalf("disposition alias = %s, %v, %v", target, found, err)
	}
	if _, _, err := NewDispositionBatch(ctx, store, closed, StatusClosed, "again"); err == nil ||
		!strings.Contains(err.Error(), "only an open finding") {
		t.Fatalf("closed finding took a disposition: %v", err)
	}
	if _, _, err := NewDispositionBatch(ctx, store, original, StatusOpen, "reopen"); err == nil {
		t.Fatal("open is not a disposition")
	}
	if _, _, err := NewDispositionBatch(ctx, store, original, StatusRefuted, "  "); err == nil {
		t.Fatal("blank resolution accepted")
	}
}

func TestDeferredDispositionIsAcknowledgedDebt(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	original, batch, err := NewTextBatch("vision tower absent", SeverityMedium,
		[]string{"internal/modelrecipe"}, []string{"no mmproj bytes"}, "owner downloads the projector", "recipe status vqa")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	deferred, deferBatch, err := NewDispositionBatch(ctx, store, original, StatusDeferred, "owner decision: projector download is out of the validation freeze")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, deferBatch); err != nil {
		t.Fatal(err)
	}
	disposition, found, err := Disposition(ctx, store, original.ID)
	if err != nil || !found || disposition.ID != deferred.ID || disposition.Status != StatusDeferred {
		t.Fatalf("disposition = %+v, %v, %v", disposition, found, err)
	}
	if _, found, err := Disposition(ctx, store, deferred.ID); err != nil || found {
		t.Fatalf("the disposition itself has a disposition: %v, %v", found, err)
	}
}
