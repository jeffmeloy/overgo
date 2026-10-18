package modelartifact

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

type inventoryCancelContext struct {
	context.Context
	checks int
	cancel context.CancelFunc
}

func (c *inventoryCancelContext) Err() error {
	c.checks++
	if c.checks == 3 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestPipelineInventoryCancellationDuringHashing(t *testing.T) {
	for _, format := range []string{"pipeline", "gguf"} {
		t.Run(format, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			observed := &inventoryCancelContext{Context: ctx, cancel: cancel}
			var inventory Inventory
			var err error
			if format == "pipeline" {
				inventory, err = FromHFPathContext(observed, pipelineFixture(t))
			} else {
				file, openErr := gguf.Open(testutil.QuantizedDenseGGUF(t, 1))
				if openErr != nil {
					t.Fatal(openErr)
				}
				defer file.Close()
				inventory, err = FromGGUFContext(observed, file, artifact.KindModel)
			}
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("hash cancellation = %v after %d checks", err, observed.checks)
			}
			if inventory.Manifest.ID.String() != "" {
				t.Fatal("cancelled inventory returned a publishable model")
			}
		})
	}
}
