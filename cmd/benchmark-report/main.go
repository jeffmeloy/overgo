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
	if strings.TrimSpace(*repository) == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		*repository = roots.Store
	}
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
	suites     map[string]suiteRow
	references []evaluation.ReferenceScore
}

// suiteRow is one suite's latest evaluation for a model.
type suiteRow struct {
	metrics    map[string]float64
	record     string
	commit     string
	wall       time.Duration
	references []evaluation.ReferenceScore
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
	// A zero bound visits the complete history; newest-first keeps the
	// latest record per suite, so every row is the current evidence.
	index := evaluation.LatestEvidence(ctx, store, names, 0)
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
			}
			for _, reference := range references {
				if reference.Suite == summary.Suite {
					suite.references = append(suite.references, reference)
				}
			}
			row.suites[summary.Suite] = suite
		}
		rows = append(rows, row)
	}
	slices.SortFunc(rows, func(a, b modelRow) int {
		return cmp.Or(cmp.Compare(a.bytes, b.bytes), strings.Compare(a.location, b.location))
	})
	return render(rows), nil
}

func render(rows []modelRow) []byte {
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
	output.WriteString("Models are the servable catalog ordered smallest first by recorded bytes. Each suite cell is the headline metric of the model's latest committed evaluation of that suite; a dash means no committed evaluation. Suites are the store's derived benchmark suites (`evaluate -list-derived-suites`), scored under the raw completion protocol unless the record says otherwise.\n\n")
	output.WriteString("## Comparison\n\n")
	output.WriteString("| Model | Bytes | Decode tok/s |")
	for _, suite := range columns {
		fmt.Fprintf(&output, " %s |", suite)
	}
	output.WriteString("\n| --- | --- | --- |")
	for range columns {
		output.WriteString(" --- |")
	}
	output.WriteString("\n")
	for _, row := range rows {
		fmt.Fprintf(&output, "| %s | %d | %s |", row.name, row.bytes, decodeCell(row))
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
		output.WriteString("| Suite | Metrics | Published reference | Commit | Wall | Record |\n| --- | --- | --- | --- | --- | --- |\n")
		for _, suite := range names {
			record := row.suites[suite]
			fmt.Fprintf(&output, "| %s | %s | %s | `%s` | %s | `%s` |\n",
				suite, metricsCell(record.metrics), referenceCell(record.references), record.commit,
				record.wall.Round(time.Millisecond), record.record)
		}
		output.WriteString("\n")
		if unmeasured := unmeasuredReferences(row); len(unmeasured) != 0 {
			fmt.Fprintf(&output, "Published references for suites without a committed evaluation: %s\n\n", referenceCell(unmeasured))
		}
	}
	output.WriteString("## Honesty\n\n")
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
