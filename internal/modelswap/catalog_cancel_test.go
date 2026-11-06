package modelswap

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// Cancel between the read-only refresh and the first memo read.
type cancelAfterRefresh struct {
	context.Context
	cancel    context.CancelCauseFunc
	refreshed bool
}

func (c *cancelAfterRefresh) Err() error {
	if c.refreshed {
		c.cancel(nil)
	}
	c.refreshed = true
	return c.Context.Err()
}

func TestCatalogCanceledMemoLoadRetries(t *testing.T) {
	root := t.TempDir()
	writer, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	testcheck.NoError(t, writer.Close())
	reader := overgodbtest.OpenReadOnly(t, root)
	resolver := CatalogResolver{Store: reader, Limit: 100}
	ctx, cancel := context.WithCancelCause(t.Context())
	defer cancel(nil)
	interrupted := &cancelAfterRefresh{Context: ctx, cancel: cancel}
	if _, _, err := resolver.Catalog(interrupted); !errors.Is(err, context.Canceled) {
		t.Fatalf("interrupted catalog: %v", err)
	}
	if resolver.memo != nil {
		t.Fatal("canceled load cached an empty memo; later requests will rehash model files")
	}
	if _, _, err := resolver.Catalog(t.Context()); err != nil || resolver.memo == nil {
		t.Fatalf("fresh request did not reload memo: %v", err)
	}
}
