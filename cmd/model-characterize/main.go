// model-characterize characterizes every servable model that RepoDB references
// -- any format -- and persists a distribution-free tensor characterization per
// model to the store. It is driven by the RepoDB catalog (discovery.Servable),
// not the filesystem: a model is fodder when RepoDB references it with an active
// inference recipe and present bytes. The measurement is lineage-bound to the
// model's existing tensor inventory, so profiles are durable and queryable by
// model.
//
//	go run ./cmd/model-characterize [-store dir] [-spectral-max-dim n]
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

func main() {
	store := flag.String("store", "", "RepoDB store directory (default: the dataroot store)")
	limit := flag.Int("limit", 4096, "max models to consider")
	samples := flag.Uint64("samples", 4096, "max sampled values per tensor")
	maxRead := flag.Uint64("max-read", 64<<20, "max bytes read across all tensors")
	spectralMaxDim := flag.Uint64("spectral-max-dim", 0, "compute effective rank for 2-D tensors whose dimensions are both within this budget (0 disables)")
	flag.Parse()
	policy := modelartifact.MeasurementPolicy{
		MaxSamplesPerTensor: *samples,
		MaxReadBytes:        *maxRead,
		SpectralMaxDim:      *spectralMaxDim,
	}
	if err := run(*store, *limit, policy); err != nil {
		fmt.Fprintf(os.Stderr, "model-characterize: %v\n", err)
		os.Exit(1)
	}
}

func run(storePath string, limit int, policy modelartifact.MeasurementPolicy) error {
	if storePath == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return fmt.Errorf("resolve data roots: %w", err)
		}
		storePath = roots.Store
	}
	store, err := repodb.Open(storePath)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}
	defer store.Close()
	count, err := characterizeCatalog(context.Background(), store, limit, policy)
	if err != nil {
		return err
	}
	fmt.Printf("characterized %d servable model(s)\n", count)
	return nil
}

// characterizeCatalog characterizes every present servable model RepoDB
// references and stores the measurement, returning the count. Per-model failures
// are reported and skipped; the catalog pass is best-effort.
func characterizeCatalog(
	ctx context.Context, store *repodb.Store, limit int, policy modelartifact.MeasurementPolicy,
) (int, error) {
	entries, err := discovery.Servable(ctx, store, limit)
	if err != nil {
		return 0, fmt.Errorf("enumerate servable models: %w", err)
	}
	count := 0
	for _, entry := range entries {
		if entry.Stale != "" {
			fmt.Fprintf(os.Stderr, "skip %s: %s\n", entry.Model, entry.Stale)
			continue
		}
		if !entry.Present {
			fmt.Fprintf(os.Stderr, "skip %s: bytes not present\n", entry.Model)
			continue
		}
		id, err := characterizeServable(ctx, store, entry, policy)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip %s: %v\n", entry.Model, err)
			continue
		}
		fmt.Printf("%s\t%s\t%s\n", id, entry.Model, entry.Location)
		count++
	}
	return count, nil
}

// characterizeServable resolves a servable model's inventory (via its active
// model definition), measures it at its recorded location (format-agnostic), and
// commits the measurement lineage-bound to that existing inventory. No
// re-registration: the model and inventory are already in the store.
func characterizeServable(
	ctx context.Context, store *repodb.Store, entry discovery.Entry, policy modelartifact.MeasurementPolicy,
) (artifact.ID, error) {
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, entry.Model, recipe.TaskInference)
	if err != nil {
		return artifact.ID{}, err
	}
	if !active {
		return artifact.ID{}, fmt.Errorf("no active inference recipe")
	}
	definitionID, ok := activation.Definition.PrimaryDependency(recipe.DependencyDefinition)
	if !ok {
		return artifact.ID{}, fmt.Errorf("active recipe has no model definition")
	}
	resolved, err := modelrecipe.ResolveModelDefinition(ctx, store, definitionID)
	if err != nil {
		return artifact.ID{}, err
	}
	document, err := modelartifact.MeasureAtLocation(resolved.Tensors, entry.Location, policy)
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := document.Batch("characterize:" + document.ID.String())
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return artifact.ID{}, fmt.Errorf("commit measurement: %w", err)
	}
	decomposition, err := modelartifact.NewComponentDecomposition(entry.Model, "", "", resolved.Tensors.Tensors)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("decompose components: %w", err)
	}
	decompositionBatch, err := decomposition.Batch("decompose:" + decomposition.ID.String())
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := store.Commit(ctx, decompositionBatch); err != nil {
		return artifact.ID{}, fmt.Errorf("commit decomposition: %w", err)
	}
	return document.ID, nil
}
