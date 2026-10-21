package server

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/gguf"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

const laneCatalogLimit = 256

type browserJourneyConfig struct {
	binary, store, model, location, multimodal string
}

// Model preparation belongs to the test that consumes it. Filtered browser
// tests neither build a model server nor inspect or mutate its catalog. The
// journey serves what the store's registrations hold, never a model named
// from the environment.
func prepareBrowserJourney(t *testing.T) (browserJourneyConfig, error) {
	t.Helper()
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		return browserJourneyConfig{}, err
	}
	if _, err := os.Stat(roots.Store); err != nil {
		return browserJourneyConfig{}, err
	}
	privateStore := filepath.Join(t.TempDir(), "store")
	if err := prepareBrowserJourneyStore(t.Context(), roots.Store, privateStore); err != nil {
		return browserJourneyConfig{}, err
	}
	servables, multimodal, err := smallestServables(t.Context(), privateStore)
	if err != nil {
		return browserJourneyConfig{}, err
	}
	if len(servables) == 0 {
		return browserJourneyConfig{}, errors.New("first-run journey: no servable inference model with bytes on disk")
	}
	binary := filepath.Join(t.TempDir(), "overgo-server.exe")
	receipt, err := processcontrol.Run(t.Context(), processcontrol.Command{
		Path: "go", Args: []string{"build", "-o", binary, "./cmd/server"}, Dir: root, Stdout: os.Stderr, Stderr: os.Stderr,
	})
	if err != nil {
		return browserJourneyConfig{}, err
	}
	if receipt.ExitCode != 0 {
		return browserJourneyConfig{}, fmt.Errorf("first-run journey: server build exited %d", receipt.ExitCode)
	}
	configured := browserJourneyConfig{binary: binary, store: privateStore, model: pickerName(servables[0]), location: servables[0]}
	if len(multimodal) > 0 {
		configured.multimodal = pickerName(multimodal[0])
	}
	return configured, nil
}

// splitShard matches a split GGUF shard's file name, which the picker shows with its directory.
var splitShard = regexp.MustCompile(`-\d+-of-\d+\.`)

// pickerName is the name the model picker shows for a model at location (boot.js modelNames): its file, and for a
// shard of a split model its directory and file, since shard names repeat across quantizations.
func pickerName(location string) string {
	base := filepath.Base(location)
	if splitShard.MatchString(base) {
		return filepath.Base(filepath.Dir(location)) + "/" + base
	}
	return base
}

// modelBytes is a model's bytes on disk: every shard its first file declares, where the first of a split model's
// shards holds only its metadata (Flash-Next's 10.9 MB of 93.7 GB).
func modelBytes(location string) (int64, error) {
	file, err := gguf.Open(location)
	if err != nil {
		return 0, err
	}
	var total int64
	for _, shard := range file.Shards() {
		total += int64(shard.Size)
	}
	return total, file.Close()
}

// smallestServables lists the store's servable inference models with bytes
// on disk, smallest model first, and among them the ones whose active
// projection recipe binds a projector with bytes on disk: the journey
// serves the cheapest model and switches to the cheapest multimodal one.
// Preparation publishes the identities it verified before releasing the
// writer to the serving child; the journey revalidates their live stats.
func smallestServables(ctx context.Context, root string) (servable, multimodal []string, err error) {
	store, err := overgodb.OpenContext(ctx, root)
	if err != nil {
		return nil, nil, errors.Join(errors.New("store did not open"), err)
	}
	defer store.Close()
	memo := discovery.LoadMemo(ctx, store)
	entries, _, err := discovery.CapabilityCatalog(ctx, store, laneCatalogLimit, memo)
	if err != nil {
		return nil, nil, err
	}
	type candidate struct {
		location   string
		size       int64
		multimodal bool
	}
	var candidates []candidate
	for _, entry := range entries {
		if !entry.Present || entry.Location == "" {
			continue
		}
		inference := false
		for _, capability := range entry.Capabilities {
			inference = inference || capability.Task == recipe.TaskInference && capability.Stale == ""
		}
		size, sizeErr := modelBytes(entry.Location)
		if !inference || sizeErr != nil {
			continue
		}
		_, projected, projectorErr := discovery.ActiveProjector(ctx, store, entry.Model, memo)
		candidates = append(candidates, candidate{location: entry.Location, size: size, multimodal: projectorErr == nil && projected})
	}
	slices.SortFunc(candidates, func(left, right candidate) int { return cmp.Compare(left.size, right.size) })
	for _, item := range candidates {
		servable = append(servable, item.location)
		if item.multimodal {
			multimodal = append(multimodal, item.location)
		}
	}
	if err := discovery.PublishMemo(ctx, store, memo); err != nil {
		return nil, nil, fmt.Errorf("persist prepared model identities: %w", err)
	}
	return servable, multimodal, nil
}
