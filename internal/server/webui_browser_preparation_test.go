package server

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
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
// tests neither build a model server nor inspect or mutate its catalog.
func prepareBrowserJourney(t *testing.T) (browserJourneyConfig, error) {
	t.Helper()
	configured := browserJourneyConfig{
		binary: os.Getenv("OVERGO_WEBUI_LANE_SERVER"), store: os.Getenv("OVERGO_WEBUI_LANE_STORE"),
		model: os.Getenv("OVERGO_WEBUI_LANE_MODEL"), location: os.Getenv("OVERGO_WEBUI_LANE_MODEL_LOCATION"),
		multimodal: os.Getenv("OVERGO_WEBUI_LANE_MULTIMODAL_MODEL"),
	}
	if configured.binary != "" || configured.store != "" || configured.model != "" || configured.location != "" {
		if configured.binary == "" || configured.store == "" || configured.model == "" || configured.location == "" {
			return browserJourneyConfig{}, errors.New("first-run journey: incomplete explicit server/store/model/location configuration")
		}
		return configured, nil
	}
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		return browserJourneyConfig{}, err
	}
	if _, err := os.Stat(roots.Store); err != nil {
		return browserJourneyConfig{}, err
	}
	servables, multimodal, err := smallestServables(t.Context(), roots.Store)
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
	configured = browserJourneyConfig{binary: binary, store: roots.Store, model: filepath.Base(servables[0]), location: servables[0]}
	if len(multimodal) > 0 {
		configured.multimodal = filepath.Base(multimodal[0])
	}
	return configured, nil
}

// smallestServables lists the store's servable inference models with bytes
// on disk, smallest file first, and among them the ones whose active
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
		info, statErr := os.Stat(entry.Location)
		if !inference || statErr != nil {
			continue
		}
		_, projected, projectorErr := discovery.ActiveProjector(ctx, store, entry.Model, memo)
		candidates = append(candidates, candidate{location: entry.Location, size: info.Size(), multimodal: projectorErr == nil && projected})
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
