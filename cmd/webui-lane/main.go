// Command webui-lane runs Overgo's required real-browser GUI acceptance
// lane: the workbench acceptance steps, the front page's keyboard, motion,
// colour and width contract, and the first-run journey against a served
// model (professional GUI campaign, gui-quality/acceptance-lane). The
// journey needs a browser, the built server binary and a servable model
// with bytes on disk in the store; a missing prerequisite reports
// UNAVAILABLE for that leg instead of failing what it cannot observe.
// With -report the lane also writes the campaign's simplification report:
// the client census at the fork beside the head and the behaviours the
// lane proved (gui-closeout/closeout).
package main

import (
	"bytes"
	"cmp"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
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
	flags := flag.NewFlagSet("webui-lane", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	report := flags.String("report", "", "write the simplification report (fork census beside head census, behaviours proven) to this path")
	fork := flags.String("fork", "", "tree measured as the campaign's fork for -report: a checkout or extracted slice holding internal/server/webui and docs/api_manifest.json")
	forkLabel := flags.String("fork-label", "", "the fork tree's commit, naming it in the report")
	headLabel := flags.String("head-label", "", "this tree's commit, naming it in the report")
	run := flags.String("run", "^"+webuilane.BrowserTestPrefix, "the browser tests to run, as go test -run takes them; a named test that skips fails the lane")
	var required []string
	flags.Func("require", "a journey line the run must write (repeatable); its absence fails the lane", func(text string) error { required = append(required, text); return nil })
	if err := flags.Parse(os.Args[1:]); err != nil || flags.NArg() != 0 || (*report != "") != (*fork != "") {
		return errors.New("usage: webui-lane [-run <pattern>] [-require <text>]... [-report <path> -fork <tree> -fork-label <commit> -head-label <commit>]")
	}
	var captured bytes.Buffer
	stdout := io.MultiWriter(os.Stdout, &captured)
	ran, err := runLane(stdout, *run)
	if err != nil {
		return err
	}
	// The verdict is the run's own output: a plan verify naming the lane gets evidence, never a silent pass.
	if ran {
		if err := webuilane.LaneVerdict(captured.String(), required); err != nil {
			return err
		}
	}
	if *report == "" {
		return nil
	}
	before, err := webuilane.MeasureTree(*fork, *forkLabel)
	if err != nil {
		return err
	}
	after, err := webuilane.MeasureTree(".", *headLabel)
	if err != nil {
		return err
	}
	proven, unobserved := webuilane.LaneObservations(captured.String())
	text := webuilane.SimplificationReport(before, after, proven, unobserved)
	if err := clioptions.WriteOutputFile(*report, []byte(text)); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "webui lane: report written to %s\n", *report)
	return nil
}

// runLane runs the browser self-check and the acceptance tests run names,
// writing the lane's observations to stdout; ran reports whether the
// tests ran at all (no browser leaves the lane UNAVAILABLE, not failed).
func runLane(stdout io.Writer, run string) (ran bool, err error) {
	browser, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		fmt.Fprintf(stdout, "webui lane: UNAVAILABLE no browser: %v\n", err)
		return false, nil
	}
	probe, err := webuilane.Open(context.Background(), browser, "data:text/html,<title>overgo-webui-lane</title>")
	if err != nil {
		return false, err
	}
	var title string
	if err := probe.SetViewport(context.Background(), len(browser), len(browser)); err == nil {
		err = probe.Evaluate(context.Background(), "document.title", &title)
	}
	_ = probe.Close()
	if err != nil {
		return false, fmt.Errorf("browser transport self-check failed: %w", err)
	}
	if title != "overgo-webui-lane" {
		return false, fmt.Errorf("browser transport self-check returned title %q", title)
	}
	env := append(os.Environ(), "OVERGO_WEBUI_LANE=1", "OVERGO_BROWSER="+browser)
	journey, unavailable := firstRunEnvironment(context.Background())
	if unavailable != "" {
		fmt.Fprintf(stdout, "webui lane: UNAVAILABLE first-run journey: %s\n", unavailable)
	} else {
		env = append(env, journey...)
	}
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path:   "go",
		Args:   []string{"test", "./internal/server", "-run", run, "-count=1", "-timeout=10m", "-v"},
		Env:    env,
		Stdout: stdout,
		Stderr: os.Stderr,
	})
	if err != nil {
		return true, err
	}
	if receipt.ExitCode != 0 {
		return true, fmt.Errorf("webui lane: acceptance exited %d", receipt.ExitCode)
	}
	fmt.Fprintf(stdout, "webui lane: PASS browser=%s\n", browser)
	return true, nil
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
	servables, multimodal, err := smallestServables(ctx, store)
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
	// The smallest model whose store declares a projector with bytes on
	// disk serves the journey's image-in leg; without one the leg proves
	// the refusal contract on the default model.
	if len(multimodal) > 0 {
		env = append(env, "OVERGO_WEBUI_LANE_MULTIMODAL_MODEL="+filepath.Base(multimodal[0]))
	}
	return env, ""
}

// smallestServables lists the store's servable inference models with bytes
// on disk, smallest file first, and among them the ones whose active
// projection recipe binds a projector with bytes on disk: the journey
// serves the cheapest model and switches to the cheapest multimodal one.
// Preparation publishes the identities it verified before releasing the
// writer to the serving child; the journey revalidates their live stats.
func smallestServables(ctx context.Context, root string) (servable, multimodal []string, err error) {
	store, err := overgodb.Open(root)
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
