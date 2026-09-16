// Command webui-lane runs selected real-browser acceptance tests.
// Compiler discovery binds ordinary selectors to their test owners.
// The first-run test prepares its own served models; layout and transport
// tests need only a browser.
package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"syscall"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
	"overgo/internal/webuilane"
)

func main() {
	clioptions.MainNamed("webui lane", run)
}

func run() error {
	flags := flag.NewFlagSet("webui-lane", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	run := flags.String("run", "^"+webuilane.BrowserTestPrefix, "the browser tests to run, as go test -run takes them; a named test that skips fails the lane")
	journeys := flags.Bool("journeys", false, "run the model journeys ("+webuilane.ModelJourneyPrefix+"*), which build, serve or hash models, instead of the page acceptances")
	screens := flags.String("screens", "", "write the captures (every tab and the picker, desktop and phone) as PNGs into this directory")
	pageURL := flags.String("url", "", "capture and audit a running server's page at this address instead of running the tests")
	var required []string
	flags.Func("require", "a journey line the run must write (repeatable); its absence fails the lane", func(text string) error { required = append(required, text); return nil })
	if err := flags.Parse(os.Args[1:]); err != nil || flags.NArg() != 0 {
		return errors.New("usage: webui-lane [-run <pattern>] [-require <text>]... [-screens <dir>] [-url <address>]")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *pageURL != "" {
		return captureLive(ctx, os.Stdout, *pageURL, *screens)
	}
	var captured bytes.Buffer
	stdout := io.MultiWriter(os.Stdout, &captured)
	var extra []string
	if *journeys {
		if *run == "^"+webuilane.BrowserTestPrefix {
			*run = "^" + webuilane.ModelJourneyPrefix
		}
		extra = append(extra, webuilane.ModelJourneyEnvironment+"=1")
	}
	if *screens != "" {
		absolute, err := filepath.Abs(*screens)
		if err != nil {
			return err
		}
		extra = append(extra, "OVERGO_WEBUI_LANE_SCREENS="+absolute)
	}
	ran, err := runLane(ctx, stdout, *run, extra)
	if err != nil {
		// The failed tests' own lines end the error, where a caller's
		// bounded tail keeps them; the whole run is kept in a file the
		// error names, since a caller keeps only that tail.
		if failures := webuilane.FailureLines(captured.String()); len(failures) > 0 {
			err = fmt.Errorf("%w: %s", err, strings.Join(failures, "; "))
		}
		if kept, keepErr := os.CreateTemp("", "webui-lane-output-*.log"); keepErr == nil {
			_, writeErr := kept.Write(captured.Bytes())
			if closeErr := kept.Close(); writeErr == nil && closeErr == nil {
				err = fmt.Errorf("%w; full output at %s", err, kept.Name())
			}
		}
		return err
	}
	// The verdict is the run's own output: a plan verify naming the lane gets evidence, never a silent pass.
	if ran {
		if err := webuilane.LaneVerdict(captured.String(), required); err != nil {
			return err
		}
	}
	return nil
}

// captureLive captures and audits a running server's page: every tab and
// the picker at each viewport, the captures written when dir is set, the
// findings listed; a finding is the error.
func captureLive(ctx context.Context, stdout io.Writer, pageURL, dir string) error {
	browser, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		return err
	}
	page, err := webuilane.Open(ctx, browser, pageURL)
	if err != nil {
		return err
	}
	defer page.Close()
	states, findings, err := webuilane.CaptureStates(ctx, page, dir)
	if err != nil {
		return err
	}
	for _, finding := range findings {
		fmt.Fprintln(stdout, finding)
	}
	fmt.Fprintln(stdout, webuilane.CaptureSummary(states, findings))
	if len(findings) > 0 {
		return fmt.Errorf("webui lane: %d layout finding(s) at %s", len(findings), pageURL)
	}
	return nil
}

// runLane runs the browser self-check and the acceptance tests run names,
// writing the lane's observations to stdout; ran reports whether the
// tests ran at all (no browser leaves the lane UNAVAILABLE, not failed).
// extra carries the screens test's capture directory and page address.
func runLane(ctx context.Context, stdout io.Writer, run string, extra []string) (ran bool, err error) {
	var listing bytes.Buffer
	if !strings.Contains(run, "/") {
		// Empty -list executes tests instead of listing them.
		listingPattern := cmp.Or(run, ".")
		args := append([]string{"test", "-list", listingPattern, "-json"}, browserTestPackages...)
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: "go", Args: args, Stdout: &listing, Stderr: os.Stderr})
		if err != nil {
			return false, err
		}
		if receipt.ExitCode != 0 {
			return false, fmt.Errorf("browser test discovery exited %d", receipt.ExitCode)
		}
	}
	packages, err := browserPackages(run, &listing)
	if err != nil {
		return false, err
	}
	browser, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		fmt.Fprintf(stdout, "webui lane: UNAVAILABLE no browser: %v\n", err)
		return false, nil
	}
	probe, err := webuilane.Open(ctx, browser, "data:text/html,<title>overgo-webui-lane</title>")
	if err != nil {
		return false, err
	}
	// The probe page renders on its own clock: the lane starts beside the
	// test groups, and a loaded host may hand back an empty title before
	// the data page has rendered, so the title is awaited, not read once.
	var title string
	if err := probe.SetViewport(ctx, len(browser), len(browser)); err == nil {
		if err = probe.Eventually(ctx, `document.title === "overgo-webui-lane"`); err == nil {
			err = probe.Evaluate(ctx, "document.title", &title)
		}
	}
	_ = probe.Close()
	if err != nil {
		return false, fmt.Errorf("browser transport self-check failed: %w", err)
	}
	if title != "overgo-webui-lane" {
		return false, fmt.Errorf("browser transport self-check returned title %q", title)
	}
	env := append(os.Environ(), "OVERGO_WEBUI_LANE=1", "OVERGO_BROWSER="+browser)
	env = append(env, extra...)
	args := append([]string{"test"}, packages...)
	args = append(args, "-run", run, "-count=1", "-timeout=20m", "-v")
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path:   "go",
		Args:   args,
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

var browserTestPackages = []string{"overgo/internal/server", "overgo/internal/webuilane", "overgo/internal/audioparity"}

// Compiler discovery selects owners, not acceptance evidence. Subtest filters
// retain the existing complete owner set until their reach is resolved.
func browserPackages(run string, listing io.Reader) ([]string, error) {
	if strings.Contains(run, "/") {
		return slices.Clone(browserTestPackages), nil
	}
	pattern, err := regexp.Compile(run)
	if err != nil {
		return nil, err
	}
	selected := map[string]bool{}
	decoder := json.NewDecoder(listing)
	for {
		var event struct {
			Package string
			Output  string
		}
		if err := decoder.Decode(&event); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, fmt.Errorf("browser test discovery: %w", err)
		}
		name := strings.TrimSpace(event.Output)
		if (strings.HasPrefix(name, webuilane.BrowserTestPrefix) || strings.HasPrefix(name, webuilane.ModelJourneyPrefix)) && pattern.MatchString(name) {
			if !slices.Contains(browserTestPackages, event.Package) {
				return nil, fmt.Errorf("browser test discovery returned unknown owner %q", event.Package)
			}
			selected[event.Package] = true
		}
	}
	var packages []string
	for _, name := range browserTestPackages {
		if selected[name] {
			packages = append(packages, name)
		}
	}
	if len(packages) == 0 {
		return nil, fmt.Errorf("browser selector %q matched no browser tests", run)
	}
	return packages, nil
}
