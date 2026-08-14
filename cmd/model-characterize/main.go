// model-characterize reads a GGUF model, computes its distribution-free tensor
// characterization (robust L-moments, order statistics, and energy descriptors),
// and persists it to a RepoDB store. The measurement is lineage-bound to the
// model's tensor inventory (itself bound to the model), so the profiles are
// durable, identity-addressed, and queryable by model rather than recomputed on
// demand.
//
//	go run ./cmd/model-characterize -store repodb-store <model.gguf>
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/repodb"
)

func main() {
	store := flag.String("store", "repodb-store", "RepoDB store directory")
	samples := flag.Uint64("samples", 4096, "max sampled values per tensor")
	maxRead := flag.Uint64("max-read", 64<<20, "max bytes read across all tensors")
	spectralMaxDim := flag.Uint64("spectral-max-dim", 0, "compute effective rank for 2-D tensors whose dimensions are both within this budget (0 disables)")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: model-characterize [-store dir] [-samples n] [-max-read bytes] [-spectral-max-dim n] <model.gguf>")
		os.Exit(2)
	}
	policy := modelartifact.MeasurementPolicy{
		MaxSamplesPerTensor: *samples,
		MaxReadBytes:        *maxRead,
		SpectralMaxDim:      *spectralMaxDim,
	}
	if err := run(*store, flag.Arg(0), policy); err != nil {
		fmt.Fprintf(os.Stderr, "model-characterize: %v\n", err)
		os.Exit(1)
	}
}

func run(storePath, modelPath string, policy modelartifact.MeasurementPolicy) error {
	store, err := repodb.Open(storePath)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}
	defer store.Close()
	file, err := gguf.Open(modelPath)
	if err != nil {
		return fmt.Errorf("open model: %w", err)
	}
	defer file.Close()
	id, err := characterizeAndStore(context.Background(), store, file, policy)
	if err != nil {
		return err
	}
	fmt.Printf("stored characterization %s\n", id)
	return nil
}

// characterizeAndStore commits the model inventory and its tensor
// characterization to the store, returning the stored measurement's identity.
// Both commits are content-addressed and idempotent: re-characterizing an
// already-registered model with the same policy is a no-op.
func characterizeAndStore(
	ctx context.Context, store *repodb.Store, file *gguf.File, policy modelartifact.MeasurementPolicy,
) (artifact.ID, error) {
	inventory, err := modelartifact.FromGGUF(file)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("build inventory: %w", err)
	}
	inventoryBatch, err := inventory.Batch("characterize-inventory:" + inventory.TensorInventory.ID.String())
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := store.Commit(ctx, inventoryBatch); err != nil {
		return artifact.ID{}, fmt.Errorf("commit inventory: %w", err)
	}
	document, err := modelartifact.MeasureGGUF(inventory.TensorInventory, file, policy)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("measure tensors: %w", err)
	}
	measurementBatch, err := document.Batch("characterize-measurement:" + document.ID.String())
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := store.Commit(ctx, measurementBatch); err != nil {
		return artifact.ID{}, fmt.Errorf("commit measurement: %w", err)
	}
	return document.ID, nil
}
