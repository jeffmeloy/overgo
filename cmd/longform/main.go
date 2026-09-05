// Command longform runs the long-form verification (owner rule
// 2026-09-04) on servable text models: the SHORT fingerprint shape, the
// LONG ladder of prompt lengths with a greedy generation and a
// teacher-forced context score per rung, the rates each rung measured,
// and a verdict against the declared floors, committed as evidence the
// suite passes read before they run; -check compares a fresh climb
// against the latest record from another inference surface.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/driver"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tokenizer"
)

// The listing bound matches the evaluation pass: the servable catalog is
// read whole, never sampled.
const catalogLimit = 256

// promptTailTokens is how much of the judged prompt's end the record
// keeps as text beside the output it continues.
const promptTailTokens = 48

// The ladder climbs while the predicted device use after the next rung
// stays under nine tenths of the device: the runtime keeps the pool's
// eighth and the driver its own share above that.
const (
	deviceFitNumerator   = uint64(9)
	deviceFitDenominator = uint64(10)
)

// deviceTotalBytes is the device's memory, read once at start; zero
// leaves the ladder unbounded by memory.
var deviceTotalBytes uint64

// corpusBytesPerToken sizes the text read for the ladder: the
// tokenizers in the catalog average under four bytes per token on the
// repository's text, so six bytes per token reads enough for the
// highest rung and its scored tail without tokenizing the whole tree.
const corpusBytesPerToken = 6

type options struct {
	Repository   string
	Root         string
	Device       int
	Publish      bool
	All          bool
	Check        bool
	OutputPrefix int
	Models       []string
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func parseOptions(args []string) (options, error) {
	flags := flag.NewFlagSet("longform", flag.ContinueOnError)
	var result options
	flags.StringVar(&result.Repository, "repo", "overgodb-store", "OvergoDB root: the servable listing, the short-prompt benchmark records, and the evidence landing")
	flags.StringVar(&result.Root, "root", ".", "repository root: the verifying commit, the inference surface, and the corpus are read from it, so a profiler that changes the working directory still runs the tool")
	flags.IntVar(&result.Device, "device", 0, "CUDA device ordinal")
	flags.BoolVar(&result.Publish, "publish", false, "commit each result as long-form evidence with a verification claim (requires a clean worktree)")
	flags.BoolVar(&result.All, "all", false, "run every servable text model, smallest first")
	flags.BoolVar(&result.Check, "check", false, "climb the ladder to the check ceiling and compare against each model's latest record from another inference surface; exit 1 on a regression")
	flags.IntVar(&result.OutputPrefix, "show", 240, "characters of the judged generation to print")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	result.Models = flags.Args()
	if result.All == (len(result.Models) > 0) {
		return options{}, errors.New("usage: longform [-repo <store>] [-device N] [-publish | -check] (-all | <model.gguf>...)")
	}
	if result.Publish && result.Check {
		return options{}, errors.New("longform: -check compares, it does not publish")
	}
	if strings.TrimSpace(result.Repository) == "" {
		return options{}, errors.New("longform: -repo is required")
	}
	return result, nil
}

// target is one model the run measures: its listing entry, the weights
// identity its claim binds to, the short-prompt rates its long-form
// rates are bounded against, and its latest long-form record when the
// store holds one.
type target struct {
	entry    discovery.Entry
	weights  artifact.ID
	bytes    int64
	short    longform.ShortRates
	record   longform.Summary
	recorded bool
}

// listTargets reads everything the run needs from the store before any
// model loads: the servable text models (a declared domain without text
// excludes a model; undeclared models are text), their sizes, the
// latest short-prompt benchmark per location, and the latest long-form
// record per location.
func listTargets(ctx context.Context, repository string, requested []string) ([]target, error) {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	memo := discovery.LoadMemo(ctx, store)
	entries, err := discovery.ServableWithMemo(ctx, store, catalogLimit, memo)
	if err != nil {
		return nil, err
	}
	benchmarks := evaluation.LatestEvidence(ctx, store, map[artifact.ID]string{}, 0).BenchmarksByLocation
	records := longform.LatestByLocation(ctx, store, 0)
	wanted := make(map[string]bool, len(requested))
	for _, path := range requested {
		wanted[longform.Key(path)] = true
	}
	selected := len(requested) > 0
	var targets []target
	for _, entry := range entries {
		if !entry.Present || entry.Stale != "" || entry.Location == "" {
			continue
		}
		if selected && !wanted[longform.Key(entry.Location)] {
			continue
		}
		domains, declared, err := evaluation.EvalDomains(ctx, store, entry.Model)
		if err != nil {
			return nil, err
		}
		if declared && !slices.Contains(domains, evaluation.DomainText) {
			if selected {
				return nil, fmt.Errorf("longform: %s declares domains %v, not a text model", entry.Location, domains)
			}
			continue
		}
		info, err := os.Stat(entry.Location)
		if err != nil {
			return nil, err
		}
		// The claim binds to the weights digest with the file location,
		// the identity the benchmark claims carry and the passes look up;
		// the catalog manifest identity carries a directory location.
		weights, err := discovery.WeightsIdentity(entry.Location, memo)
		if err != nil {
			return nil, err
		}
		benchmark := benchmarks[entry.Location]
		record, recorded := records[longform.Key(entry.Location)]
		targets = append(targets, target{
			entry: entry, weights: weights, bytes: info.Size(),
			short: longform.ShortRates{
				PromptTokensPerSecond: benchmark.PromptTokensPerSecond,
				DecodeTokensPerSecond: benchmark.DecodeTokensPerSecond,
			},
			record: record, recorded: recorded,
		})
		delete(wanted, longform.Key(entry.Location))
	}
	if len(wanted) > 0 {
		return nil, fmt.Errorf("longform: not servable model locations: %s", strings.Join(slices.Sorted(maps.Keys(wanted)), ", "))
	}
	if len(targets) == 0 {
		return nil, errors.New("longform: the store holds no servable text model")
	}
	// Smallest first (owner rule 2026-09-01): the cheap models surface a
	// broken run in seconds; the model whose run costs minutes goes last.
	slices.SortFunc(targets, func(a, b target) int {
		if a.bytes != b.bytes {
			return int(a.bytes - b.bytes)
		}
		return strings.Compare(a.entry.Location, b.entry.Location)
	})
	return targets, nil
}

func run(args []string, output io.Writer) error {
	ctx := context.Background()
	options, err := parseOptions(args)
	if err != nil {
		return err
	}
	// Publishing binds the evidence to the verifying commit; a dry run
	// still names the commit it ran on when the tree is clean.
	commit, commitErr := runrecord.VerifyingCommit(options.Root)
	if options.Publish && commitErr != nil {
		return commitErr
	}
	surface, err := longform.Surface(ctx, options.Root)
	if err != nil {
		return err
	}
	targets, err := listTargets(ctx, options.Repository, options.Models)
	if err != nil {
		return err
	}
	floors := longform.DeclaredFloors()
	ceiling := 0
	if options.Check {
		ceiling = floors.CheckRungCeiling
	}
	if cuda, err := driver.Open(); err == nil {
		if err := cuda.Init(); err == nil {
			if info, err := cuda.DeviceInfo(options.Device); err == nil {
				deviceTotalBytes = info.TotalMemoryBytes
			}
		}
		cuda.Close()
	}
	var failures []error
	for index, target := range targets {
		name := filepath.Base(target.entry.Location)
		fmt.Fprintf(output, "long-form %d/%d: %s\n", index+1, len(targets), target.entry.Location)
		if options.Check {
			switch {
			case !target.recorded:
				failures = append(failures, fmt.Errorf("%s: no long-form record to check against; run: go run ./cmd/longform -publish %q", name, target.entry.Location))
				fmt.Fprintln(output, "  FAIL: no record")
				continue
			case target.record.Result.Surface == surface:
				fmt.Fprintf(output, "  PASS: the record measured this inference surface (%.12s)\n", surface)
				continue
			}
		}
		result, err := measure(ctx, output, options, target, commit, surface, floors, ceiling)
		if err != nil {
			failures = append(failures, fmt.Errorf("%s: %w", name, err))
			fmt.Fprintf(output, "  ERROR %v\n", err)
			continue
		}
		report(output, result, options.OutputPrefix)
		if options.Check {
			verdict := longform.Compare(target.record.Result, result, floors, ceiling)
			fmt.Fprintf(output, "  against record %.12s (surface %.12s): %s\n", target.record.Record, target.record.Result.Surface, verdict)
			if !verdict.Passed {
				failures = append(failures, fmt.Errorf("%s: regression: %s", name, verdict))
			}
			continue
		}
		if !result.Verdict.Passed {
			failures = append(failures, fmt.Errorf("%s: %s", name, result.Verdict))
		}
		if options.Publish {
			record, err := publish(ctx, options.Repository, target.weights, result)
			if err != nil {
				return err
			}
			fmt.Fprintf(output, "  evidence committed: record=%s surface=%.12s commit=%.12s\n", record, surface, commit)
		}
	}
	return errors.Join(failures...)
}

// measure loads one model, runs the SHORT shape and the ladder, and
// closes it before anything else touches the store: the runner holds
// the device and the store's reader for the whole run.
func measure(ctx context.Context, output io.Writer, options options, target target, commit, surface string, floors longform.Floors, ceiling int) (longform.Result, error) {
	runner, err := clioptions.OpenRunner(ctx, options.Repository, target.entry.Location, clioptions.BuildOpenOptions(options.Device, nil, 1))
	if err != nil {
		return longform.Result{}, err
	}
	defer runner.Close()
	properties := runner.ModelProperties()
	highest := int(properties.ContextLength)
	if ceiling > 0 {
		highest = min(highest, ceiling)
	}
	text, err := longform.Corpus(options.Root, corpusBytesPerToken*(highest+floors.ScoreTokens))
	if err != nil {
		return longform.Result{}, err
	}
	corpus, err := runner.TokenizeText(text, true, false)
	if err != nil {
		return longform.Result{}, err
	}
	rungs := longform.LadderRungs(properties.ContextLength, len(corpus), floors, ceiling)
	if len(rungs) == 0 {
		return longform.Result{}, fmt.Errorf("longform: no ladder rung fits a context of %d tokens and a corpus of %d", properties.ContextLength, len(corpus))
	}
	// Each stage reports as it completes, with the device memory it left
	// allocated, so a run that stalls or grows names its stage.
	stage := func(name string) {
		memory, err := runner.DeviceMemoryStats(ctx)
		if err != nil {
			fmt.Fprintf(output, "  stage %s (device memory unread: %v)\n", name, err)
			return
		}
		fmt.Fprintf(output, "  stage %s: device %d MiB (peak %d MiB)\n", name, memory.CurrentBytes>>20, memory.PeakBytes>>20)
	}
	stage(fmt.Sprintf("loaded, %d corpus tokens, rungs %v", len(corpus), rungs))
	if err := longform.Warm(ctx, runner, corpus[:rungs[0]]); err != nil {
		return longform.Result{}, err
	}
	stage("warm")
	shape, err := longform.Short(ctx, runner, corpus, floors)
	if err != nil {
		return longform.Result{}, err
	}
	stage("short")
	// A rung doubles the cache the last rung added; the ladder stops
	// when that growth would not fit the device's memory, since a rung
	// past it thrashed the E4B at its 65536 rung with the device full.
	previous := uint64(0)
	climbed, stop, err := longform.Ladder(ctx, runner, corpus, rungs, floors, func(rung longform.Rung) string {
		measure := rung.Measure
		stage(fmt.Sprintf("rung %d: prompt %.1f tok/s, decode %.1f tok/s, gain %.3f", measure.PromptTokens,
			measure.PromptTokensPerSecond, measure.DecodeTokensPerSecond, measure.Score.ContextGain))
		memory, err := runner.DeviceMemoryStats(ctx)
		if err != nil || deviceTotalBytes == 0 {
			return ""
		}
		growth := uint64(0)
		if previous != 0 && memory.CurrentBytes > previous {
			growth = memory.CurrentBytes - previous
		}
		previous = memory.CurrentBytes
		if next := memory.CurrentBytes + 2*growth; next > deviceTotalBytes/deviceFitDenominator*deviceFitNumerator {
			return fmt.Sprintf("the device holds %d MiB of %d and the next rung would take it to %d", memory.CurrentBytes>>20, deviceTotalBytes>>20, next>>20)
		}
		return ""
	})
	if err != nil {
		return longform.Result{}, err
	}
	judged := climbed[len(climbed)-1]
	for _, rung := range climbed {
		if rung.Measure.PromptTokens == floors.PromptTokens {
			judged = rung
		}
	}
	prompt := corpus[:judged.Measure.PromptTokens]
	tail, err := runner.Detokenize(prompt[max(0, len(prompt)-promptTailTokens):], inference.RenderText)
	if err != nil {
		return longform.Result{}, err
	}
	outputText, err := runner.Detokenize(dtype.ConvertSlice[tokenizer.TokenID](judged.OutputIDs), inference.RenderText)
	if err != nil {
		return longform.Result{}, err
	}
	result := longform.Result{
		ModelPath: properties.Path, ModelName: properties.Name, Architecture: properties.Architecture, FileType: properties.FileType,
		Commit: commit, Surface: surface, PromptSource: "README.md, docs, internal (see longform.Corpus)",
		Measure: judged.Measure, Short: target.short, Floors: floors,
		Shape: shape, Rungs: climbed, LadderStop: stop,
		PromptTail: tail, Output: outputText,
	}
	result.Verdict = longform.Judge(result.Measure, result.Short, floors)
	return result, nil
}

func publish(ctx context.Context, repository string, model artifact.ID, result longform.Result) (artifact.ID, error) {
	store, err := overgodb.Open(repository)
	if err != nil {
		return artifact.ID{}, err
	}
	record, publishErr := longform.Publish(ctx, store, model, result)
	return record, errors.Join(publishErr, store.Close())
}

// report prints the short shape, every rung, the judged verdict, and
// the head of the judged output so a reader sees what the model wrote,
// not only how fast.
func report(output io.Writer, result longform.Result, prefix int) {
	fmt.Fprintf(output, "  short %d->%d tok: NLL %.3f, decode %.1f tok/s\n",
		result.Shape.PromptTokens, len(result.Shape.OutputIDs), result.Shape.NLL, result.Shape.Measure.DecodeTokensPerSecond)
	for _, rung := range result.Rungs {
		measure := rung.Measure
		fmt.Fprintf(output, "  rung %d: prompt %.1f tok/s, decode %d tok at %.1f tok/s, distinct 4-gram %.3f, span %d, NLL %.3f long / %.3f short (gain %.3f)\n",
			measure.PromptTokens, measure.PromptTokensPerSecond, measure.OutputTokens, measure.DecodeTokensPerSecond,
			measure.DistinctFourGramRatio, measure.LongestRepeatedSpan,
			measure.Score.LongContextNLL, measure.Score.ShortContextNLL, measure.Score.ContextGain)
		cost := measure.Execution
		fmt.Fprintf(output, "    per token: %.0f launches, %.1f syncs, %.0f B up, %.0f B down, %.2f graph instantiations, %.1f graph launches; prompt: %d launches, %d instantiations, %d syncs, %d B up\n",
			cost.KernelLaunchesPerToken, cost.SynchronizationsPerToken, cost.HostToDeviceBytesPerToken, cost.DeviceToHostBytesPerToken,
			cost.GraphInstantiationsPerToken, cost.GraphLaunchesPerToken,
			cost.PromptKernelLaunches, cost.PromptGraphInstantiations, cost.PromptStreamSynchronizations, cost.PromptHostToDeviceBytes)
	}
	if result.LadderStop != "" {
		fmt.Fprintf(output, "  ladder stopped: %s\n", result.LadderStop)
	}
	measure := result.Measure
	stopped := ""
	if measure.StoppedEarly {
		stopped = " (stopped early)"
	}
	fmt.Fprintf(output, "  judged rung %d: prompt %.1f tok/s (short-prompt %.1f); decode %d tok at %.1f tok/s (short-prompt %.1f)%s -> %s\n",
		measure.PromptTokens, measure.PromptTokensPerSecond, result.Short.PromptTokensPerSecond,
		measure.OutputTokens, measure.DecodeTokensPerSecond, result.Short.DecodeTokensPerSecond, stopped, result.Verdict)
	fmt.Fprintf(output, "  prompt tail: ...%s\n", strings.ReplaceAll(result.PromptTail, "\n", " "))
	text := strings.ReplaceAll(result.Output, "\n", " ")
	if runes := []rune(text); len(runes) > prefix {
		text = string(runes[:prefix]) + "..."
	}
	fmt.Fprintf(output, "  output: %s\n", text)
}
