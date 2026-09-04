// Command longform runs the long-form verification (owner rule
// 2026-09-04) on servable text models: one long prompt, one greedy
// generation, the prompt and decode rates, the degeneration measures,
// and a verdict against the declared floors, committed as evidence the
// suite passes read before they run.
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
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// The prompt source is the repository's own README: prose long enough
// for every model's prompt cut, in the tree at every commit, and read by
// the same tool at the same path on every run.
const defaultPromptFile = "README.md"

// The listing bound matches the evaluation pass: the servable catalog is
// read whole, never sampled.
const catalogLimit = 256

// promptTailTokens is how much of the prompt's end the record keeps as
// text beside the output it continues.
const promptTailTokens = 48

type options struct {
	Repository   string
	Device       int
	PromptFile   string
	Publish      bool
	All          bool
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
	flags.IntVar(&result.Device, "device", 0, "CUDA device ordinal")
	flags.StringVar(&result.PromptFile, "prompt-file", defaultPromptFile, "text the long prompt is cut from")
	flags.BoolVar(&result.Publish, "publish", false, "commit each result as long-form evidence with a verification claim (requires a clean worktree)")
	flags.BoolVar(&result.All, "all", false, "run every servable text model, smallest first")
	flags.IntVar(&result.OutputPrefix, "show", 240, "characters of each generation to print")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	result.Models = flags.Args()
	if result.All == (len(result.Models) > 0) {
		return options{}, errors.New("usage: longform [-repo <store>] [-device N] [-prompt-file <text>] [-publish] (-all | <model.gguf>...)")
	}
	if strings.TrimSpace(result.Repository) == "" {
		return options{}, errors.New("longform: -repo is required")
	}
	return result, nil
}

// target is one model the run measures: its listing entry and the
// short-prompt rates its long-form rates are bounded against.
type target struct {
	entry discovery.Entry
	bytes int64
	short longform.ShortRates
}

// listTargets reads everything the run needs from the store before any
// model loads: the servable text models (a declared domain without text
// excludes a model; undeclared models are text), their sizes, and the
// latest short-prompt benchmark per location.
func listTargets(ctx context.Context, repository string, requested []string) ([]target, error) {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	entries, err := discovery.ServableWithMemo(ctx, store, catalogLimit, discovery.LoadMemo(ctx, store))
	if err != nil {
		return nil, err
	}
	benchmarks := evaluation.LatestEvidence(ctx, store, map[artifact.ID]string{}, 0).BenchmarksByLocation
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
		benchmark := benchmarks[entry.Location]
		targets = append(targets, target{
			entry: entry, bytes: info.Size(),
			short: longform.ShortRates{
				PromptTokensPerSecond: benchmark.PromptTokensPerSecond,
				DecodeTokensPerSecond: benchmark.DecodeTokensPerSecond,
			},
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
	commit, commitErr := runrecord.VerifyingCommit(".")
	if options.Publish && commitErr != nil {
		return commitErr
	}
	surface, err := longform.Surface(ctx, ".")
	if err != nil {
		return err
	}
	promptText, err := os.ReadFile(options.PromptFile)
	if err != nil {
		return err
	}
	targets, err := listTargets(ctx, options.Repository, options.Models)
	if err != nil {
		return err
	}
	floors := longform.DeclaredFloors()
	var failures []error
	for index, target := range targets {
		fmt.Fprintf(output, "long-form %d/%d: %s\n", index+1, len(targets), target.entry.Location)
		result, err := measure(ctx, options, target, string(promptText), commit, surface, floors)
		if err != nil {
			failures = append(failures, fmt.Errorf("%s: %w", filepath.Base(target.entry.Location), err))
			fmt.Fprintf(output, "  ERROR %v\n", err)
			continue
		}
		report(output, result, options.OutputPrefix)
		if !result.Verdict.Passed {
			failures = append(failures, fmt.Errorf("%s: %s", filepath.Base(target.entry.Location), result.Verdict))
		}
		if options.Publish {
			record, err := publish(ctx, options.Repository, target.entry.Model, result)
			if err != nil {
				return err
			}
			fmt.Fprintf(output, "  evidence committed: record=%s surface=%.12s commit=%.12s\n", record, surface, commit)
		}
	}
	return errors.Join(failures...)
}

// measure loads one model, runs it, and closes it before anything else
// touches the store: the runner holds the device and the store's reader
// for the whole run.
func measure(ctx context.Context, options options, target target, promptText, commit, surface string, floors longform.Floors) (longform.Result, error) {
	runner, err := clioptions.OpenRunner(ctx, options.Repository, target.entry.Location, clioptions.BuildOpenOptions(options.Device, nil, 1))
	if err != nil {
		return longform.Result{}, err
	}
	defer runner.Close()
	prompt, continuation, err := longform.Cut(runner, promptText, floors.PromptTokens, floors.ScoreTokens)
	if err != nil {
		return longform.Result{}, err
	}
	if err := longform.Warm(ctx, runner, prompt); err != nil {
		return longform.Result{}, err
	}
	generation, err := longform.Run(ctx, runner, prompt, floors.OutputTokens)
	if err != nil {
		return longform.Result{}, err
	}
	generation.Measure.Score, err = longform.Score(ctx, runner, prompt, continuation, floors.ShortContextTokens)
	if err != nil {
		return longform.Result{}, err
	}
	tail, err := runner.Detokenize(prompt[max(0, len(prompt)-promptTailTokens):], inference.RenderText)
	if err != nil {
		return longform.Result{}, err
	}
	properties := runner.ModelProperties()
	result := longform.Result{
		ModelPath: properties.Path, ModelName: properties.Name, Architecture: properties.Architecture, FileType: properties.FileType,
		Commit: commit, Surface: surface, PromptSource: filepath.ToSlash(options.PromptFile),
		Measure: generation.Measure, Short: target.short, Floors: floors, PromptTail: tail, Output: generation.Text,
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

// report prints the measure, the verdict, and the head of the output so
// a reader sees what the model wrote, not only how fast.
func report(output io.Writer, result longform.Result, prefix int) {
	measure := result.Measure
	stopped := ""
	if measure.StoppedEarly {
		stopped = " (stopped early)"
	}
	fmt.Fprintf(output, "  prompt %d tok at %.1f tok/s (short-prompt %.1f); decode %d tok at %.1f tok/s (short-prompt %.1f)%s\n",
		measure.PromptTokens, measure.PromptTokensPerSecond, result.Short.PromptTokensPerSecond,
		measure.OutputTokens, measure.DecodeTokensPerSecond, result.Short.DecodeTokensPerSecond, stopped)
	fmt.Fprintf(output, "  distinct 4-gram ratio %.3f, longest repeated span %d tok; true continuation NLL %.3f long / %.3f short nat/token (gain %.3f) -> %s\n",
		measure.DistinctFourGramRatio, measure.LongestRepeatedSpan,
		measure.Score.LongContextNLL, measure.Score.ShortContextNLL, measure.Score.ContextGain, result.Verdict)
	fmt.Fprintf(output, "  prompt tail: ...%s\n", strings.ReplaceAll(result.PromptTail, "\n", " "))
	text := strings.ReplaceAll(result.Output, "\n", " ")
	if runes := []rune(text); len(runes) > prefix {
		text = string(runes[:prefix]) + "..."
	}
	fmt.Fprintf(output, "  output: %s\n", text)
}
