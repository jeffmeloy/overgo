package landing

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io"
	"io/fs"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"overgo/internal/cuda/driver"
	"overgo/internal/devicescope"
	"overgo/internal/gitauthority"
	"overgo/internal/gosource"
	"overgo/internal/processcontrol"
	"overgo/internal/testskip"
)

// lane is one check that drives real hardware or a browser. It runs when the
// change reaches the test closure of its packages, or touches a path under
// one of its prefixes; go list, not a manifest, decides the closure.
type lane struct {
	name     string
	packages []string
	prefixes []string
	// tags are the lane's build context: go lists its closure and runs its
	// command, after the command's verb, with them.
	tags    string
	command func(changed string) []string
	// environment is added to the gate's own for the lane's command.
	environment []string
	// hostOnly says the lane needs no device, a browser against a server on the
	// host: it runs beside the lanes that share the device, not after them.
	hostOnly bool
	// tests is the lane's own package that holds its tests, and testPrefix
	// what their names begin with: a path that enters the lane only through
	// what that package imports reaches just the tests whose files' own
	// imports take it in, and the lane runs those alone.
	tests, testPrefix string
	// binary says the lane runs its packages' built binaries, not their
	// tests: its reach is their build closure, so a test file, the test
	// support only tests import and a test's data reach it nowhere.
	binary bool
	// find finds the lane's packages in the checkout, where packages does not name them, and the lane runs only those
	// of them whose test closure the change enters, named after its command (reach.packages).
	find func(repo string) ([]string, error)
}

// reach is what a change reaches a lane by: the first changed path that
// reached it, with the lane's own package whose closure that path entered,
// empty when none did, and the lane's tests it runs alone, none when it runs
// whole.
type reach struct {
	by    string
	tests []string
	// packages are the lane's packages whose test closure the change entered, for a lane that runs those alone (find).
	packages []string
}

// splitLanes divides the lanes a change reached into those that need no device,
// which run at once beside the rest, and those that share it, which run one after
// another in their order: the wall of the lanes is then the device lanes' alone
// when a host lane is shorter than they are.
func splitLanes(reached []lane) (beside, shared []lane) {
	for _, candidate := range reached {
		if candidate.hostOnly {
			beside = append(beside, candidate)
		} else {
			shared = append(shared, candidate)
		}
	}
	return beside, shared
}

var lanes = []lane{
	{
		// The device lane runs its scopes' tests as its own processes, not as
		// imports, so its reach is the closure of those packages. Its runner's
		// own files reach it, and what the runner alone imports does not: that
		// cannot change a device result, and the runner's tests hold the
		// commands it builds.
		name: "device", packages: devicePackages(), prefixes: []string{"kernels/", "cmd/device-lane/"},
		command: func(changed string) []string { return []string{"run", "./cmd/device-lane", "-paths-file", changed} },
	},
	{
		// The page's assets, browser tests, lane and page declarations. Its
		// runner's own files reach it, as a prefix, and what the runner alone
		// imports does not.
		name: "webui", packages: []string{"./internal/server", "./internal/webuilane"},
		prefixes: []string{"internal/server/webui/", "internal/server/webui_", "cmd/webui-lane/", "internal/webuilane/",
			"internal/server/workspace_manifest.json", "internal/server/workspace_schema.json"},
		command:  func(string) []string { return []string{"run", "./cmd/webui-lane"} },
		hostOnly: true,
		// The server imports every model runtime, and of its browser tests only
		// those whose files take one in call it: a media change runs those
		// alone. The prefix is webuilane.BrowserTestPrefix, which a test holds;
		// the landing does not import the browser's driver.
		tests: "internal/server", testPrefix: "TestWebUIBrowser",
	},
	{
		// The CUDA suites outside the device lane's scopes that build their
		// own inputs: programs on synthetic weights, the trainers and the
		// kernels only they call, found in the checkout (cudaLanePackages).
		// One package at a time shares the device, and only those the change
		// enters run: a failing test of one it does not enter refused it,
		// since go test's cache replays a pass and never a failure (the
		// owner's review, 2026-11-14: image-readings refused by
		// trainingworkflow's sampler test).
		name: "cuda", find: cudaLanePackages,
		command:     func(string) []string { return []string{"test", "-json", "-p=1", "-timeout=0"} },
		environment: []string{cudaTestEnvironment},
	},
}

// CycleTags are the build tags of the tests only the improvement cycle runs,
// those that open real models or read the live store. No gate lane builds
// with them, so a landing never compiles those tests; vet and fix read with
// them (checkedTags), so every tagged file is still checked.
const CycleTags = "modeltest,integration"

// pairTag is the build tag of the paired held-out comparisons, each a fast path's answers against its reference's over
// the held-out parts a measurement run asks. No lane builds with it: a comparison is the check of a change to the path
// it compares, named in that step's verify, and the cycle's cells judge the served path's held-out quality every
// cycle. In the cycle the four took 237 of the 304 minutes its model suites ran on 2026-10-31, answered whole again
// each cycle, since the inference code they build on changes in nearly every one.
const pairTag = "pairtest"

// checkedTags are the tags vet and fix read the tree with, so every tagged file is still checked.
const checkedTags = CycleTags + "," + pairTag + "," + wholeTreeTag

// cycleLanes are what the improvement cycle runs in the gate's place: real
// models, the live store and the served-model journeys, which guard what
// the cycle compares. validate -run runs each of them every cycle, and a lane
// that runs go test runs it with -count=1, since the models and references
// it reads lie outside the module and go test's cache does not see them
// change (SQA-315); the cycle reports its pass.
var cycleLanes = []lane{
	{
		// A served-model journey or the server command.
		name:     "model-journey",
		packages: []string{"./internal/modelswap", "./internal/libraryintake", "./internal/discovery", "./cmd/server"},
		prefixes: []string{"cmd/server/", "internal/server/webui_browser_firstrun", "internal/server/webui_browser_library_validation",
			"internal/server/webui_browser_preparation", "internal/server/webui_browser_store_test.go", "internal/server/webui_served_projector",
			// The microphone journey lives in audioparity with the fixtures
			// it shares, so any file there reaches it.
			"internal/audioparity/"},
		command: func(string) []string { return []string{"run", "./cmd/webui-lane", "-journeys"} },
	},
	{
		// The audio acceptances run the recorded speech models against the
		// reference store; the landing's go test skips them.
		// cmd/evaluate and cmd/recipe are the producers the acceptances
		// build and run as their own processes.
		name: "audio", packages: []string{"./internal/audioparity", "./cmd/evaluate", "./cmd/recipe"},
		tags: CycleTags,
		// Device acceptance executes against external weights and kernels each cycle.
		command:     func(string) []string { return []string{"test", "-count=1", "./internal/audioparity"} },
		environment: []string{testskip.StoreAcceptanceEnv + "=1", cudaTestEnvironment},
	},
	{
		// The real-model suites: tests behind the cycle's tags, and the
		// packages whose device tests all open real models. One package at a
		// time shares the device; adaptiveparity alone runs about 11
		// minutes, past go test's default timeout.
		name: "models", packages: modelPackages, tags: CycleTags,
		command: func(string) []string {
			return append([]string{"test", "-count=1", "-p=1", "-timeout=0"}, modelPackages...)
		},
		// The frozen Krea and SenseNova generations, Krea's runtime baselines and the Fractale leaderships run here,
		// behind the variables their records name, where they had run nowhere (opt-in-tests/run-or-remove).
		environment: []string{cudaTestEnvironment, "OVERGO_WAN_FULL=1", "OVERGO_KREA_BASELINE=1", "OVERGO_SENSENOVA_FULL=1", "OVERGO_FRACTALE_BASELINE=1"},
	},
	{
		name: "needle-reference", packages: []string{"./internal/seq2seq"}, tags: CycleTags,
		command: func(string) []string {
			return []string{"test", "-count=1", "-timeout=0", "./internal/seq2seq", "-run", "^TestNeedleCallsMatchUpstreamOnEvaluationCases$"}
		},
		environment: []string{cudaTestEnvironment},
	},
	{
		// The long generated sequences: the store's lifecycles at crashLongSeeds, minutes a commit's checks do not
		// spend, held to the promises a commit holds a few dozen to.
		name: "sequences", packages: []string{"./internal/overgodb"}, tags: CycleTags,
		command: func(string) []string {
			return []string{"test", "-count=1", "-timeout=0", "./internal/overgodb", "-run", "^TestLongStoreLifecycles$"}
		},
	},
}

// cudaTestEnvironment opts a lane's tests into the real device; without it
// cudatest.Require skips them.
const cudaTestEnvironment = "OVERGO_CUDA_TEST=1"

// cudaRequirement is the call a real-device test opens with, split so that
// source quoting it is not read as calling it.
const cudaRequirement = "cudatest" + ".Require("

// cudaLanePackages are the packages the cuda lane runs: each package under internal/ or cmd/ one of whose test files,
// as this host builds them (go/build's MatchFile, its GOOS and constraints; a file behind CycleTags is none), calls
// cudatest.Require, but those the device lane's scopes run. The walk that held a hand-kept list complete (the
// cuda-lanes rule) now is the list (the owner's review, 2026-11-14).
func cudaLanePackages(repo string) ([]string, error) {
	device := lane{packages: devicePackages()}
	found := map[string]bool{}
	for _, top := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(repo, top), func(name string, entry fs.DirEntry, err error) error {
			// A repository without the top directory holds no tests there.
			if errors.Is(err, fs.ErrNotExist) && name == filepath.Join(repo, top) {
				return fs.SkipDir
			}
			if err != nil || entry.IsDir() || !strings.HasSuffix(name, "_test.go") {
				return err
			}
			relative, err := filepath.Rel(repo, filepath.Dir(name))
			directory := filepath.ToSlash(relative)
			if err != nil || found[directory] || device.owns(directory) {
				return err
			}
			if built, err := build.Default.MatchFile(filepath.Dir(name), entry.Name()); err != nil || !built {
				return err
			}
			data, err := os.ReadFile(name)
			found[directory] = err == nil && bytes.Contains(data, []byte(cudaRequirement))
			return err
		})
		if err != nil {
			return nil, err
		}
	}
	var packages []string
	for _, directory := range slices.Sorted(maps.Keys(found)) {
		if found[directory] {
			packages = append(packages, "./"+directory)
		}
	}
	return packages, nil
}

// lanesOf are the gate's lanes as the checkout at repo holds them: each lane that finds its packages there (find) with
// them found.
func lanesOf(repo string) ([]lane, error) {
	found := slices.Clone(lanes)
	for index, candidate := range found {
		if candidate.find == nil {
			continue
		}
		packages, err := candidate.find(repo)
		if err != nil {
			return nil, err
		}
		found[index].packages = packages
	}
	return found, nil
}

// modelPackages are the packages whose real-model tests the cycle's models
// lane runs.
var modelPackages = []string{
	"./internal/thoughtbank", "./internal/latentvideo", "./internal/quant", "./internal/diffusionimage", "./internal/latentimage",
	"./internal/mediacapability", "./internal/sensenovarecipe", "./internal/routedlm", "./internal/inference", "./internal/adaptiveparity",
	"./internal/composition", "./internal/speechsynth", "./internal/tabularicl", "./internal/modelintake", "./internal/server",
	"./cmd/evaluate", "./cmd/longform", "./cmd/recipe", "./cmd/train", "./cmd/validate", "./cmd/benchmark-report", "./cmd/loop",
	"./internal/evaluation", "./internal/dataset", "./internal/rae", "./internal/vitencoder", "./internal/seriesforecast", "./internal/mosstrain",
	"./internal/densecausaltrain", "./internal/modeldevice", "./internal/oscillatorimage", "./internal/hybridtrain", "./internal/cuda/nvvfx",
}

// laneEnvironment is the environment a lane's command runs in: the gate's own, what the lane adds, and the CUDA
// driver the host runs, which cudatest.Require reads so go test keys a cached CUDA pass on its driver.
func laneEnvironment(candidate lane) []string {
	return append(append(os.Environ(), candidate.environment...), driver.IdentityEnvironment+"="+driver.LibraryIdentity())
}

// CycleLane is one lane of the improvement cycle: the go command's arguments
// that run it and the environment it adds.
type CycleLane struct {
	Name        string
	Arguments   []string
	Environment []string
}

// CycleLanes lists the improvement cycle's lanes, which run every cycle.
func CycleLanes() []CycleLane {
	var listed []CycleLane
	for _, candidate := range cycleLanes {
		listed = append(listed, CycleLane{Name: candidate.name, Arguments: candidate.arguments(""), Environment: candidate.environment})
	}
	return listed
}

// arguments run the lane, its build tags after the command's verb.
func (candidate lane) arguments(changed string) []string {
	return slices.Insert(candidate.command(changed), 1, "-tags", candidate.tags)
}

// laneTurn names the turn a landing holds while its lanes that use the card run, and a measurement run while its
// phases on the card run (HoldLaneTurn): the card's own turn (driver.CardTurnOf), which every context is created in, so
// the processes of those lanes and phases ride on the holder's place, and a render beside them gives the card up once
// for them all, where each test had waited out a picture (card-handover/lane-turn); a test names its own.
var laneTurn = func() (string, error) { return driver.CardTurnOf(0) }

// takeLaneTurn takes the lane turn for purpose, saying on output, once, whom it waits for and how long it waited.
func takeLaneTurn(purpose string, output io.Writer) (func() error, error) {
	name, err := laneTurn()
	if err != nil {
		return nil, err
	}
	return processcontrol.TakeTurn(name, purpose, output)
}

// HoldLaneTurn runs work in the host's lane turn: a measurement run's phases that use the device hold it, so a landing
// beside the run runs its host stages and its lanes wait for the device in code, where every build, test and commit
// was once held for the run by a written rule.
func HoldLaneTurn(work func() error) error {
	release, err := takeLaneTurn("a measurement run's phases on the card", os.Stderr)
	if err != nil {
		return err
	}
	return errors.Join(work(), release())
}

// letGo frees the checkout while the lanes wait for a lane turn another holds, a measurement run's above all: on
// 2026-10-30 a landing waited on the run's turn for hours, and as a landing ships the checkout's dirty tree, no other
// row could be edited until it landed. The stages read the candidate in a worktree of its own (checkOut), and the
// commit publishes it whatever the checkout holds by then (settled), so only the commit turn is left to give up. A
// merge keeps its commit turn. It says whether it gave the commit turn up.
func (l *landing) letGo() (bool, error) {
	if len(l.merging) > 0 {
		return false, nil
	}
	// The wait is the device's and no other commit's, so the commit turn goes free for it: on 2026-10-30 a report-only
	// commit waited hours behind a landing that was itself waiting on a measurement run. It goes free whether or not
	// the card is held as the landing comes to take it: a card found free could be taken before the landing took it,
	// and the landing would then wait for the card holding the commit turn (SQA-261).
	release := l.turn
	if release != nil {
		l.turn = nil
		if err := release(); err != nil {
			return true, err
		}
	}
	fmt.Fprintf(l.Log, "land: the landing takes the card's turn on the change as it stood, in %s: the checkout is free\n", l.tree)
	return release != nil, nil
}

// takeTurnBack takes the commit turn again after the card's wait gave it up. A commit may have landed meanwhile, and
// the change was brought up to the tree as it was, so a trunk that moved refuses the landing rather than check and
// publish onto what it was not brought up to. It lands again on the trunk as it stands.
func (l *landing) takeTurnBack(ctx context.Context) error {
	turn, err := l.holdTurn(ctx)
	if err != nil {
		return err
	}
	l.turn = turn
	tip, err := gitauthority.Query(ctx, l.Repo, "rev-parse", cmp.Or(l.onto, "HEAD"))
	if err != nil {
		return err
	}
	if moved := strings.TrimSpace(string(tip)); moved != l.builtOn() {
		return fmt.Errorf("land: %.12s landed while this waited for the device, and the checks passed on %.12s; land again", moved, l.builtOn())
	}
	return nil
}

// holdCard takes the card's turn for the rest of the landing, which gives it up as it ends (Run): every process it starts
// from then rides on its place, the lanes' and the scorecard's benchmark alike, where each had taken the card per
// context and waited out a render's clip, and the
// benchmark timed its wait as its load (card-stages/whole-landing). It waits with the checkout let go (letGo) and no
// side of the quiet machine held, which it takes again once the turn is its own: a timed cell of a run that holds the
// turn takes the machine alone, and would otherwise wait on a landing that waits on the run. The wait is said on the
// landing's log as it starts and measured here.
func (l *landing) holdCard(ctx context.Context) error {
	if l.card != nil {
		return nil
	}
	freed, err := l.letGo()
	if err != nil {
		return err
	}
	machine := l.quietMachine
	if l.quiet != nil {
		released := l.quiet()
		l.quiet = nil
		if released != nil {
			return released
		}
	}
	waiting := time.Now()
	release, err := takeLaneTurn("the commit checks of "+l.Ref, l.Log)
	if err != nil {
		return err
	}
	l.card, l.spent.Turn = release, time.Since(waiting).Round(time.Millisecond)
	if machine != "" {
		quiet, err := l.holdQuiet(ctx, machine)
		if err != nil {
			return err
		}
		l.quiet = quiet
	}
	// The commit turn is taken back once the card is the landing's, so the checkout is free for the wait alone: a
	// commit another session made while the landing ran its stages refused it after them, 30 minutes of checks in
	// (process-identity/start, 2026-11-12), where one made during the wait refuses it now, before them.
	if !freed {
		return nil
	}
	return l.takeTurnBack(ctx)
}

// reachesCard says whether the change reaches a lane that shares the device, so the landing holds the card's turn from
// its first stage on the card (holdCard). A landing that reaches none takes no turn, so a measurement run's phases
// keep it waiting for nothing it uses: on 2026-10-31 a landing whose one lane was webui waited 7.99 h for a run's turn
// (e63fd568a).
func (l *landing) reachesCard() (bool, error) {
	candidates, err := lanesOf(l.stageRoot())
	if err != nil {
		return false, err
	}
	reaches, err := discover(l.stageRoot(), candidates...)
	if err != nil {
		return false, err
	}
	for _, candidate := range candidates {
		if !candidate.hostOnly && reaches(candidate, l.paths).by != "" {
			return true, nil
		}
	}
	return false, nil
}

// inTurnFor runs work in the card's turn when one of shared needs the device, the resource the turn guards, taking it
// if the landing holds it not yet (holdCard). Lanes that need none run at once.
func (l *landing) inTurnFor(ctx context.Context, shared []lane, work func() error) error {
	if len(shared) == 0 {
		return work()
	}
	if err := l.holdCard(ctx); err != nil {
		return err
	}
	return work()
}

// lanes runs every lane the change reaches: the lanes that share the device one
// after another, and the lanes that need none beside them. A change that reaches a
// lane of the device runs them in the host's lane turn, so another landing on this
// machine waits for it rather than running its lanes beside these on the one GPU;
// one that reaches none takes no turn (inTurnFor). A turn, not a claim on the device: a
// claim would make every lane process its member and erase the process
// identities the device's own admission tests exercise. The measurement says how
// long finding the reached lanes and waiting for the turn took, and how each lane
// ended (runLanes).
func (l *landing) lanes(ctx context.Context) error {
	changed, err := os.CreateTemp("", "overgo-landing-paths-*.txt")
	if err != nil {
		return err
	}
	defer os.Remove(changed.Name())
	if _, err := changed.WriteString(strings.Join(l.paths, "\n")); err != nil {
		changed.Close()
		return err
	}
	if err := changed.Close(); err != nil {
		return err
	}
	var reached []lane
	var skipped []string
	started := time.Now()
	candidates, err := lanesOf(l.stageRoot())
	if err != nil {
		return err
	}
	reaches, err := discover(l.stageRoot(), candidates...)
	if err != nil {
		return err
	}
	pulled, named, entered := map[string]string{}, map[string][]string{}, map[string][]string{}
	for _, candidate := range candidates {
		if got := reaches(candidate, l.paths); got.by != "" {
			reached, pulled[candidate.name], named[candidate.name], entered[candidate.name] = append(reached, candidate), got.by, got.tests, got.packages
			if len(got.tests) > 0 {
				pulled[candidate.name] += fmt.Sprintf(", %d of its tests", len(got.tests))
			}
			if candidate.find != nil {
				pulled[candidate.name] += fmt.Sprintf(", %d of its %d packages", len(got.packages), len(candidate.packages))
			}
		} else {
			skipped = append(skipped, candidate.name)
		}
	}
	discovery := time.Since(started).Round(time.Millisecond)
	if len(reached) == 0 {
		l.details["lanes"] = describeLanes(discovery, 0, nil, skipped)
		return nil
	}
	beside, shared := splitLanes(reached)
	// The card's turn is the landing's from its first stage on the card (holdCard), and its wait is measured there.
	return l.inTurnFor(ctx, shared, func() error {
		turnWait := l.spent.Turn
		entries, err := l.runLanes(ctx, beside, shared, func(ctx context.Context, candidate lane) (string, error) {
			arguments := append(candidate.arguments(changed.Name()), entered[candidate.name]...)
			if tests := named[candidate.name]; len(tests) > 0 {
				arguments = append(arguments, "-run", "^("+strings.Join(tests, "|")+")$")
			}
			return l.runIn(ctx, l.stageRoot(), laneEnvironment(candidate), "go", arguments...)
		})
		if err != nil {
			return err
		}
		var passed []string
		for _, candidate := range reached {
			passed = append(passed, entries[candidate.name]+" for "+pulled[candidate.name])
		}
		l.details["lanes"] = describeLanes(discovery, turnWait, passed, skipped)
		return nil
	})
}

// runLanes runs, by run, the lanes beside the device's at once and the device's one after another, and says how each
// passed lane ended. The first lane to fail refuses the landing, by the tests it failed, and cancels the others, which
// are not its cause: the web UI lane's failed test once cancelled the device lane, and the refusal named the device
// lane's cancellation (lane-cause/do).
func (l *landing) runLanes(ctx context.Context, beside, shared []lane, run func(context.Context, lane) (string, error)) (map[string]string, error) {
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	entries := map[string]string{}
	var mu sync.Mutex
	each := func(candidate lane) {
		fmt.Fprintf(l.Log, "land: lane %s runs\n", candidate.name)
		began := time.Now()
		output, err := run(ctx, candidate)
		wall := time.Since(began).Round(time.Millisecond)
		mu.Lock()
		defer mu.Unlock()
		switch {
		case err == nil:
			entries[candidate.name] = describeLane(candidate.name, wall, output)
		case ctx.Err() != nil:
			fmt.Fprintf(l.Log, "land: lane %s cancelled after %s\n", candidate.name, wall)
		default:
			fmt.Fprintf(l.Log, "land: lane %s refused after %s\n", candidate.name, wall)
			cancel(fmt.Errorf("lane %s: %w", candidate.name, testFailure(output, err)))
		}
	}
	var group sync.WaitGroup
	for _, candidate := range beside {
		group.Go(func() { each(candidate) })
	}
	for _, candidate := range shared {
		if ctx.Err() != nil {
			break
		}
		each(candidate)
	}
	group.Wait()
	return entries, context.Cause(ctx)
}

// passedPackage opens go test's line for a package that passed, in its
// plain output and in the output events of -json alike.
const passedPackage = "ok  \t"

// describeLane is a passed lane's entry in the measurement: its wall time
// and, where its output carries go test's package lines, how many packages go
// test executed and how many it replayed from cache. Its processes ride on the
// landing's card turn, whose wait the lanes' measurement says once.
func describeLane(name string, wall time.Duration, output string) string {
	executed, cached := 0, 0
	for line := range strings.Lines(output) {
		var event struct{ Action, Output string }
		if json.Unmarshal([]byte(line), &event) == nil && event.Action == "output" {
			line = event.Output
		}
		switch {
		case !strings.HasPrefix(line, passedPackage):
		case strings.Contains(line, cachedMarker):
			cached++
		default:
			executed++
		}
	}
	entry := fmt.Sprintf("%s passed %s", name, wall)
	if executed+cached > 0 {
		entry += fmt.Sprintf(" executed %d cached %d", executed, cached)
	}
	return entry
}

// describeLanes is the lanes stage's measurement: the time finding the
// reached lanes took, the wait for the card's turn when a lane ran,
// each passed lane's entry, and the lanes the change did not reach.
func describeLanes(discovery, turnWait time.Duration, passed, skipped []string) string {
	entries := []string{"discovery " + discovery.String()}
	if len(passed) > 0 {
		entries = append(entries, "card wait "+turnWait.String())
	}
	entries = append(entries, passed...)
	if len(skipped) > 0 {
		entries = append(entries, "skipped "+strings.Join(skipped, " "))
	}
	return strings.Join(entries, ", ")
}

// discover lists the lanes' test closures once per build context -- lanes
// share a listing only when they build with the same tags -- and returns
// whether changed paths reach a lane: a module file reaches every lane, a
// path under one of its prefixes reaches it, and so does a path in its test
// closure, what its own packages' test binaries import, or such a package
// imports without tests; a test file behind tags the lane does not build,
// such as CycleTags, reaches no lane that leaves it out. A package or
// dependency a listing cannot load refuses the landing. What it returns for a
// lane is its reach: a lane that runs for a change names what pulled it in, and
// one whose tests are named runs alone those a path reaching it only through
// their package's imports calls.
func discover(repo string, candidates ...lane) (func(lane, []string) reach, error) {
	patterns := map[string][]string{}
	for _, candidate := range candidates {
		for _, pattern := range candidate.packages {
			// A lane package this checkout does not hold, as in a fixture
			// repository, reaches nothing; one it holds must load.
			base, _ := strings.CutSuffix(strings.TrimPrefix(pattern, "./"), "/...")
			if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(base))); err == nil {
				patterns[candidate.tags] = append(patterns[candidate.tags], pattern)
			}
		}
	}
	selections := map[string]gosource.BuildSelection{}
	for tags, packages := range patterns {
		selection, err := gosource.HostBuildClosure(repo, append([]string{"-tags", tags, "-test"}, packages...)...)
		if err != nil {
			return nil, err
		}
		selections[tags] = selection
	}
	directories := map[string]map[string]string{}
	for tags, selection := range selections {
		directories[tags] = map[string]string{}
		for file, importPath := range selection.Packages {
			directories[tags][importPath] = path.Dir(file)
		}
	}
	calls := map[string]map[string][]string{}
	for _, candidate := range candidates {
		if candidate.tests != "" {
			called, err := calledTests(repo, candidate, selections[candidate.tags], directories[candidate.tags])
			if err != nil {
				return nil, err
			}
			calls[candidate.name] = called
		}
	}
	return func(candidate lane, paths []string) reach {
		selection, directories, closure := selections[candidate.tags], directories[candidate.tags], map[string]string{}
		// owners are, for each closure directory, every one of the lane's packages that takes it in.
		owners := map[string][]string{}
		// The listing holds every lane of the build context, so the lane's
		// closure is taken from its own packages: each one, what it imports,
		// and what its test binary imports, each directory kept with the
		// lane's package that took it in, in the listing's order. A
		// dependency compiled for a test binary is listed as "path [binary]";
		// one outside the repository has no directory.
		for _, file := range slices.Sorted(maps.Keys(selection.Packages)) {
			importPath := selection.Packages[file]
			if !candidate.owns(path.Dir(file)) || candidate.binary && strings.HasSuffix(file, "_test.go") {
				continue
			}
			imports, tested := selection.Dependencies[importPath], selection.Dependencies[importPath+".test"]
			if candidate.binary {
				// A test variant of the package imports what its tests do; the
				// binary builds the package alone.
				imports, tested = selection.BuildDependencies[importPath], nil
			}
			for _, dependency := range slices.Concat([]string{importPath}, imports, tested) {
				dependency, _, _ = strings.Cut(dependency, " [")
				if _, held := closure[directories[dependency]]; !held {
					closure[directories[dependency]] = path.Dir(file)
				}
				if owned := owners[directories[dependency]]; !slices.Contains(owned, path.Dir(file)) {
					owners[directories[dependency]] = append(owned, path.Dir(file))
				}
			}
		}
		// enters is what a changed path reaches the lane by, with the closure
		// directory it entered and the lane's package that took that one in;
		// by is empty when the path reaches nothing.
		enters := func(name string) (by, entered, owner string) {
			if name == "go.mod" || name == "go.sum" || slices.ContainsFunc(candidate.prefixes, func(prefix string) bool { return strings.HasPrefix(name, prefix) }) {
				return name, "", ""
			}
			// A test file compiles only into its own package's test binary,
			// so it reaches the lane only when that package is the lane's and
			// the lane's build compiles it; a deleted one no longer says
			// which builds did, so its package's lane is reached.
			if candidate.binary && (strings.HasSuffix(name, "_test.go") || slices.Contains(strings.Split(path.Dir(name), "/"), "testdata")) {
				return "", "", ""
			}
			if strings.HasSuffix(name, "_test.go") {
				_, absent := os.Stat(filepath.Join(repo, filepath.FromSlash(name)))
				if candidate.owns(path.Dir(name)) && (selection.Files[name] || errors.Is(absent, fs.ErrNotExist)) {
					return name, "", ""
				}
				return "", "", ""
			}
			// A data file nested under a closure package, testdata or an
			// embedded recipe, is that package's input; a Go file belongs to
			// its own package, so only data walks up.
			for parent := path.Dir(name); ; parent = path.Dir(parent) {
				if owner, held := closure[parent]; held {
					if owner == parent {
						return name, parent, owner
					}
					return name + " through " + owner, parent, owner
				}
				if strings.HasSuffix(name, ".go") || parent == "." {
					return "", "", ""
				}
			}
		}
		called, narrowed := calls[candidate.name], reach{}
		if candidate.find != nil {
			// A lane that runs only the packages a change enters names them: every package taking in the directory a path
			// entered, a test file's own package, and all of them for a path that reaches the lane whole.
			entered := map[string]bool{}
			for _, name := range paths {
				by, directory, _ := enters(name)
				switch {
				case by == "":
					continue
				case strings.HasSuffix(name, "_test.go"):
					entered["./"+path.Dir(name)] = true
				case directory == "":
					for _, pattern := range candidate.packages {
						entered[pattern] = true
					}
				default:
					for _, owner := range owners[directory] {
						entered["./"+owner] = true
					}
				}
				narrowed.by = cmp.Or(narrowed.by, by)
			}
			narrowed.packages = slices.Sorted(maps.Keys(entered))
			return narrowed
		}
		for _, name := range paths {
			by, entered, owner := enters(name)
			switch {
			case by == "":
			case called == nil || owner != candidate.tests || entered == owner:
				return reach{by: by}
			case len(called[entered]) > 0:
				narrowed.by = cmp.Or(narrowed.by, by)
				narrowed.tests = append(narrowed.tests, called[entered]...)
			}
		}
		slices.Sort(narrowed.tests)
		narrowed.tests = slices.Compact(narrowed.tests)
		return narrowed
	}, nil
}

// calledTests lists, for each directory a test file of the lane's tests
// package takes in through its own imports, the lane's tests that file holds.
func calledTests(repo string, candidate lane, selection gosource.BuildSelection, directories map[string]string) (map[string][]string, error) {
	called := map[string][]string{}
	for _, file := range slices.Sorted(maps.Keys(selection.Files)) {
		if path.Dir(file) != candidate.tests || !strings.HasSuffix(file, "_test.go") || !selection.Files[file] {
			continue
		}
		parsed, err := parser.ParseFile(token.NewFileSet(), filepath.Join(repo, filepath.FromSlash(file)), nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		var tests []string
		for _, declaration := range parsed.Decls {
			if function, ok := declaration.(*ast.FuncDecl); ok && function.Recv == nil && strings.HasPrefix(function.Name.Name, candidate.testPrefix) {
				tests = append(tests, function.Name.Name)
			}
		}
		taken := map[string]bool{}
		for _, spec := range parsed.Imports {
			imported, _ := strconv.Unquote(spec.Path.Value)
			for _, dependency := range append([]string{imported}, selection.Dependencies[imported]...) {
				if directory, held := directories[dependency]; held && len(tests) > 0 && !taken[directory] {
					taken[directory] = true
					called[directory] = append(called[directory], tests...)
				}
			}
		}
	}
	return called, nil
}

// owns reports whether one of the lane's package patterns, ./directory or
// ./directory/..., names the directory.
func (candidate lane) owns(directory string) bool {
	return slices.ContainsFunc(candidate.packages, func(pattern string) bool {
		base, recursive := strings.CutSuffix(strings.TrimPrefix(pattern, "./"), "/...")
		return directory == base || recursive && strings.HasPrefix(directory, base+"/")
	})
}

// devicePackages are the device lane's scopes as package patterns; the cuda
// tree is recursive.
func devicePackages() []string {
	var patterns []string
	for _, packagePath := range devicescope.Packages() {
		if packagePath == "internal/cuda" {
			packagePath += "/..."
		}
		patterns = append(patterns, "./"+packagePath)
	}
	return patterns
}
