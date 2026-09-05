// Command webui-lane runs Overgo's required real-browser GUI acceptance
// lane: the workbench acceptance steps, the front page's keyboard, motion,
// colour and width contract, and the first-run journey against a served
// model (professional GUI campaign, gui-quality/acceptance-lane). The
// journey needs a browser, the built server binary and a servable model
// with bytes on disk in the store; a missing prerequisite reports
// UNAVAILABLE for that leg instead of failing what it cannot observe.
package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/webuilane"
)

const laneCatalogLimit = 256

func main() {
	clioptions.MainNamed("webui lane", run)
}

func run() error {
	browser, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		fmt.Printf("webui lane: UNAVAILABLE no browser: %v\n", err)
		return nil
	}
	probe, err := webuilane.Open(context.Background(), browser, "data:text/html,<title>overgo-webui-lane</title>")
	if err != nil {
		return err
	}
	var title string
	if err := probe.SetViewport(context.Background(), len(browser), len(browser)); err == nil {
		err = probe.Evaluate(context.Background(), "document.title", &title)
	}
	_ = probe.Close()
	if err != nil {
		return fmt.Errorf("browser transport self-check failed: %w", err)
	}
	if title != "overgo-webui-lane" {
		return fmt.Errorf("browser transport self-check returned title %q", title)
	}
	env := append(os.Environ(), "OVERGO_WEBUI_LANE=1", "OVERGO_BROWSER="+browser)
	journey, unavailable := firstRunEnvironment(context.Background())
	if unavailable != "" {
		fmt.Printf("webui lane: UNAVAILABLE first-run journey: %s\n", unavailable)
	} else {
		env = append(env, journey...)
	}
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path:   "go",
		Args:   []string{"test", "./internal/server", "-run", "^TestWebUIBrowser", "-count=1", "-timeout=6m", "-v"},
		Env:    env,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	})
	if err != nil {
		return err
	}
	if receipt.ExitCode != 0 {
		return fmt.Errorf("webui lane: acceptance exited %d", receipt.ExitCode)
	}
	fmt.Printf("webui lane: PASS browser=%s\n", browser)
	return nil
}

// firstRunEnvironment prepares the served-model journey: the server binary
// built from this tree and the smallest servable inference model the store
// holds with bytes on disk; the journey switches through the picker itself. The reason a prerequisite is missing is returned as text.
func firstRunEnvironment(ctx context.Context) ([]string, string) {
	// The store comes from the data-root contract (OVERGO_DATA_ROOT, local-models.json,
	// or ./overgodb-store), so the lane finds it from a gate's candidate tree too.
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return nil, err.Error()
	}
	store := roots.Store
	if _, err := os.Stat(store); err != nil {
		return nil, "no store at " + store
	}
	scratch, err := os.MkdirTemp("", "webui-lane-*")
	if err != nil {
		return nil, err.Error()
	}
	binary := filepath.Join(scratch, "overgo-server.exe")
	if _, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"build", "-o", binary, "./cmd/server"}, Stdout: os.Stderr, Stderr: os.Stderr,
	}); err != nil {
		return nil, "server binary did not build: " + err.Error()
	}
	servables, err := smallestServables(ctx, store)
	if err != nil {
		return nil, err.Error()
	}
	if len(servables) == 0 {
		return nil, "no servable inference model with bytes on disk"
	}
	env := []string{
		"OVERGO_WEBUI_LANE_SERVER=" + binary, "OVERGO_WEBUI_LANE_STORE=" + store,
		"OVERGO_WEBUI_LANE_MODEL=" + filepath.Base(servables[0]), "OVERGO_WEBUI_LANE_MODEL_LOCATION=" + servables[0],
	}
	return env, ""
}

// smallestServables lists the store's servable inference models with bytes
// on disk, smallest file first: the journey serves the cheapest model.
func smallestServables(ctx context.Context, root string) ([]string, error) {
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return nil, errors.Join(errors.New("store did not open"), err)
	}
	defer store.Close()
	entries, _, err := discovery.CapabilityCatalog(ctx, store, laneCatalogLimit, discovery.LoadMemo(ctx, store))
	if err != nil {
		return nil, err
	}
	type candidate struct {
		location string
		size     int64
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
		candidates = append(candidates, candidate{location: entry.Location, size: info.Size()})
	}
	slices.SortFunc(candidates, func(left, right candidate) int { return cmp.Compare(left.size, right.size) })
	locations := make([]string, 0, len(candidates))
	for _, item := range candidates {
		locations = append(locations, item.location)
	}
	return locations, nil
}
