// benchmark-report renders docs/BENCHMARK.md from the store's committed
// evidence: every servable model, smallest first, against every derived
// suite it has scored, with the code commit and wall time of each record,
// beside its decode-throughput benchmark claim. The document is generated,
// never edited: -check fails when the file differs from the store, -update
// rewrites it. Published reference scores enter as store declarations and
// print beside the measured value, so a comparison against a model card is a
// recorded fact, not prose.
package main

import (
	"cmp"
	"context"
	"errors"
	"flag"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

const reportPath = "docs/BENCHMARK.md"

func main() {
	clioptions.MainNamed("benchmark-report", func() error { return run(os.Args[1:]) })
}

// metricFormat prints a metric with the precision the sources report:
// accuracies to the case, decode rates to a tenth of a token.
const metricFormat = "%.4g"

func run(args []string) error {
	flags := flag.NewFlagSet("benchmark-report", flag.ContinueOnError)
	repository := flags.String("repo", "", "OvergoDB root; empty resolves via the data-root contract (OVERGO_DATA_ROOT, local-models.json, or ./overgodb-store)")
	check := flags.Bool("check", false, "fail when docs/BENCHMARK.md differs from the store's evidence")
	update := flags.Bool("update", false, "rewrite docs/BENCHMARK.md from the store's evidence")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *check == *update {
		return errors.New("usage: benchmark-report [-check|-update] [-repo <store>]")
	}
	resolved, err := dataroot.StoreRoot(*repository)
	if err != nil {
		return err
	}
	*repository = resolved
	data, err := generate(context.Background(), *repository)
	if err != nil {
		return err
	}
	return clioptions.OutputGenerated(
		data, reportPath, *check, *update,
		"docs/BENCHMARK.md is stale; regenerate with: go run ./cmd/benchmark-report -update",
		os.Stdout,
	)
}

// modelRow is one servable model with its committed evidence.
type modelRow struct {
	name       string
	location   string
	bytes      int64
	benchmark  evaluation.BenchmarkSummary
	measured   bool
	longForm   longform.Summary
	verified   bool
	suites     map[string]suiteRow
	references []evaluation.ReferenceScore
}

// suiteRow is one suite's latest evaluation for a model.
type suiteRow struct {
	metrics    map[string]float64
	record     string
	commit     string
	wall       time.Duration
	cases      uint64
	references []evaluation.ReferenceScore
}

// perCaseCell reports the wall time one case took on average; a suite
// whose case count is not in the catalog shows the gap.
func perCaseCell(record suiteRow) string {
	if record.cases == 0 || record.wall == 0 {
		return "—"
	}
	return (record.wall / time.Duration(record.cases)).Round(time.Millisecond).String()
}

func generate(ctx context.Context, repository string) ([]byte, error) {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	// The report's denominator is the whole servable catalog: the
	// listing bound is the widest the query accepts, never a sample.
	entries, err := discovery.ServableWithMemo(ctx, store, math.MaxInt, discovery.LoadMemo(ctx, store))
	if err != nil {
		return nil, err
	}
	names := evaluation.DerivedSuiteNames(ctx, store, evaluation.ListingAuthorities())
	descriptors := evaluation.DerivedSuiteDescriptors(ctx, store, evaluation.ListingAuthorities())
	// A zero bound visits the complete history; newest-first keeps the
	// latest record per suite, so every row is the current evidence.
	index := evaluation.LatestEvidence(ctx, store, names, 0)
	longForms := longform.LatestByLocation(ctx, store, 0)
	rows := make([]modelRow, 0, len(entries))
	for _, entry := range entries {
		if !entry.Present || entry.Stale != "" || entry.Location == "" {
			continue
		}
		info, err := os.Stat(entry.Location)
		if err != nil {
			return nil, err
		}
		row := modelRow{
			name:     strings.TrimSuffix(filepath.Base(entry.Location), filepath.Ext(entry.Location)),
			location: entry.Location, bytes: info.Size(), suites: map[string]suiteRow{},
		}
		row.benchmark, row.measured = index.BenchmarksByLocation[entry.Location]
		row.longForm, row.verified = longForms[longform.Key(entry.Location)]
		references, err := evaluation.LoadReferenceScores(ctx, store, entry.Model)
		if err != nil {
			return nil, err
		}
		row.references = references
		for _, summary := range index.EvaluationsByRecipe[entry.Recipe] {
			record, err := runrecord.RequireEvaluation(ctx, store, summary.Record)
			if err != nil {
				return nil, err
			}
			bound, err := runrecord.RequireRun(ctx, store, record.Run)
			if err != nil {
				return nil, err
			}
			suite := suiteRow{
				metrics: summary.Metrics, record: summary.Record.String(),
				commit: bound.CodeCommit, wall: time.Duration(bound.MeasuredNS),
				cases: descriptors[summary.Suite].Cases,
			}
			for _, reference := range references {
				if reference.Suite == summary.Suite {
					suite.references = append(suite.references, reference)
				}
			}
			// One column per suite and prompting protocol: the raw
			// anchor and the template-shaped pass stand side by side.
			row.suites[summary.Suite+" ("+summary.Prompting+")"] = suite
		}
		rows = append(rows, row)
	}
	slices.SortFunc(rows, func(a, b modelRow) int {
		return cmp.Or(cmp.Compare(a.bytes, b.bytes), strings.Compare(a.location, b.location))
	})
	return render(rows, descriptors), nil
}

// renderSuites states each catalog suite's shape and the two protocols,
// so a headline number is read against its case count and shot count
// rather than against a published full-benchmark figure.
func renderSuites(output *strings.Builder, descriptors map[string]evaluation.SuiteDescriptor) {
	output.WriteString("## Suites\n\n")
	output.WriteString("Every suite is scored 0-shot from the store's catalog. The raw-completion protocol scores each candidate's likelihood after the prompt; the chat-template protocol renders the prompt through the model's declared template, generates a short answer, and reads the answer letter. Published references are declared per model with their own protocol and shot count, so a cell and its reference are comparable only when those match.\n\n")
	output.WriteString("| Suite | Kind | Cases |\n| --- | --- | --- |\n")
	for _, source := range slices.Sorted(maps.Keys(descriptors)) {
		descriptor := descriptors[source]
		fmt.Fprintf(output, "| %s | %s | %d |\n", source, descriptor.Kind, descriptor.Cases)
	}
	output.WriteString("\n")
}

func render(rows []modelRow, descriptors map[string]evaluation.SuiteDescriptor) []byte {
	suites := map[string]bool{}
	for _, row := range rows {
		for suite := range row.suites {
			suites[suite] = true
		}
	}
	columns := slices.Sorted(maps.Keys(suites))
	var output strings.Builder
	output.WriteString("# Benchmark report\n\n")
	output.WriteString("Generated from the store's committed evaluation and benchmark records. Do not edit this file directly; regenerate with `go run ./cmd/benchmark-report -update`.\n\n")
	output.WriteString("Models are the servable catalog ordered smallest first by recorded bytes. Each suite cell is the headline metric of the model's latest committed evaluation of that suite under one prompting protocol; a dash means no committed evaluation. Suites are the store's derived benchmark suites (`evaluate -list-derived-suites`). The raw-completion protocol scores the suite's prompt text unchanged and is the recorded anchor comparable with lm-eval. The chat-template protocol wraps the prompt in the model's declared template with thinking disabled, asks for the letter only, greedily generates the model's answer, and reads the first standalone candidate letter from it; an answer naming no candidate counts as incorrect. Model cards report generative results, so the chat-template rows are the ones to set beside a published number.\n\n")
	renderSuites(&output, descriptors)
	output.WriteString("## Comparison\n\n")
	output.WriteString("| Model | Bytes | Prompt tok/s | Decode tok/s | Long prompt tok/s | Long decode tok/s | Long-form |")
	for _, suite := range columns {
		fmt.Fprintf(&output, " %s |", suite)
	}
	output.WriteString("\n| --- | --- | --- | --- | --- | --- | --- |")
	for range columns {
		output.WriteString(" --- |")
	}
	output.WriteString("\n")
	for _, row := range rows {
		fmt.Fprintf(&output, "| %s | %d | %s | %s | %s | %s | %s |", row.name, row.bytes, promptCell(row), decodeCell(row),
			longPromptCell(row), longDecodeCell(row), longFormCell(row))
		for _, suite := range columns {
			output.WriteString(" " + headlineCell(row.suites[suite]) + " |")
		}
		output.WriteString("\n")
	}
	output.WriteString("\n## Records\n\n")
	for _, row := range rows {
		fmt.Fprintf(&output, "### %s\n\n", row.name)
		fmt.Fprintf(&output, "Location `%s`, %d bytes.\n\n", filepath.ToSlash(row.location), row.bytes)
		if len(row.suites) == 0 {
			output.WriteString("No committed evaluation.\n\n")
			continue
		}
		names := slices.Sorted(maps.Keys(row.suites))
		output.WriteString("| Suite | Metrics | Published reference | Commit | Cases | Wall | Per case | Record |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n")
		for _, suite := range names {
			record := row.suites[suite]
			fmt.Fprintf(&output, "| %s | %s | %s | `%s` | %d | %s | %s | `%s` |\n",
				suite, metricsCell(record.metrics), referenceCell(record.references), record.commit,
				record.cases, record.wall.Round(time.Millisecond), perCaseCell(record), record.record)
		}
		output.WriteString("\n")
		if unmeasured := unmeasuredReferences(row); len(unmeasured) != 0 {
			fmt.Fprintf(&output, "Published references for suites without a committed evaluation: %s\n\n", referenceCell(unmeasured))
		}
	}
	output.WriteString("## Audit\n\n")
	evaluated, referenced := 0, 0
	for _, row := range rows {
		if len(row.suites) != 0 {
			evaluated++
		}
		for _, suite := range row.suites {
			if len(suite.references) != 0 {
				referenced++
				break
			}
		}
	}
	fmt.Fprintf(&output, "%d servable model(s); %d with committed evaluations; %d with a published reference declared; %d suite(s) with evidence.\n",
		len(rows), evaluated, referenced, len(columns))
	output.WriteString("Models without evidence appear with dashes rather than being omitted. A published reference is a store declaration naming its source and protocol; the measured value is not adjusted toward it.\n")
	return []byte(output.String())
}

func decodeCell(row modelRow) string {
	if !row.measured || row.benchmark.DecodeTokensPerSecond == 0 {
		return "—"
	}
	return fmt.Sprintf(metricFormat, row.benchmark.DecodeTokensPerSecond)
}

// The long-form cells report the committed long-form verification: the
// prompt and decode rates at the long prompt, and the verdict with the
// output measures and the commit it ran on, so a suite score is read
// beside the long-input behaviour the pass was admitted on.
func longPromptCell(row modelRow) string {
	if !row.verified || row.longForm.Result.Measure.PromptTokensPerSecond == 0 {
		return "—"
	}
	return fmt.Sprintf(metricFormat, row.longForm.Result.Measure.PromptTokensPerSecond)
}

func longDecodeCell(row modelRow) string {
	if !row.verified || row.longForm.Result.Measure.DecodeTokensPerSecond == 0 {
		return "—"
	}
	return fmt.Sprintf(metricFormat, row.longForm.Result.Measure.DecodeTokensPerSecond)
}

func longFormCell(row modelRow) string {
	if !row.verified {
		return "—"
	}
	result := row.longForm.Result
	verdict := "PASS"
	if !result.Verdict.Passed {
		verdict = "FAIL"
	}
	return fmt.Sprintf("%s (%d→%d tok, context gain %+.2f nat, distinct 4-gram %.2f, span %d, `%.12s`)", verdict,
		result.Measure.PromptTokens, result.Measure.OutputTokens, result.Measure.Score.ContextGain,
		result.Measure.DistinctFourGramRatio, result.Measure.LongestRepeatedSpan, result.Commit)
}

// promptCell reports the committed benchmark's prompt-processing rate;
// a record from before the summary carried it, or no record at all,
// shows the gap rather than a number.
func promptCell(row modelRow) string {
	if !row.measured || row.benchmark.PromptTokensPerSecond == 0 {
		return "—"
	}
	return fmt.Sprintf(metricFormat, row.benchmark.PromptTokensPerSecond)
}

// headlineMetrics orders the metric a suite cell shows: an accuracy
// when the suite scores one, else the suite's own aggregate.
var headlineMetrics = []string{"accuracy", "prompt_strict", "perplexity", "mean"}

func headlineCell(record suiteRow) string {
	if record.metrics == nil {
		return "—"
	}
	for _, name := range headlineMetrics {
		if value, found := record.metrics[name]; found {
			return name + " " + fmt.Sprintf(metricFormat, value)
		}
	}
	names := slices.Sorted(maps.Keys(record.metrics))
	if len(names) == 0 {
		return "—"
	}
	return names[0] + " " + fmt.Sprintf(metricFormat, record.metrics[names[0]])
}

func metricsCell(metrics map[string]float64) string {
	names := slices.Sorted(maps.Keys(metrics))
	parts := make([]string, 0, len(names))
	for _, name := range names {
		parts = append(parts, name+"="+fmt.Sprintf(metricFormat, metrics[name]))
	}
	return strings.Join(parts, "<br>")
}

func referenceCell(references []evaluation.ReferenceScore) string {
	if len(references) == 0 {
		return "—"
	}
	parts := make([]string, 0, len(references))
	for _, reference := range references {
		parts = append(parts, fmt.Sprintf("%s %s=%s (%s; %s)",
			reference.Suite, reference.Metric, fmt.Sprintf(metricFormat, reference.Value), reference.Protocol, reference.Source))
	}
	return strings.Join(parts, "<br>")
}

// unmeasuredReferences lists the model's declared references whose
// suite has no committed evaluation, so a declaration never hides
// behind a missing measurement.
func unmeasuredReferences(row modelRow) []evaluation.ReferenceScore {
	var unmeasured []evaluation.ReferenceScore
	for _, reference := range row.references {
		if _, measured := row.suites[reference.Suite]; !measured {
			unmeasured = append(unmeasured, reference)
		}
	}
	return unmeasured
}
