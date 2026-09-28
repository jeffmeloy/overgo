package main

import (
	"context"
	"fmt"
	"go/scanner"
	"go/token"
	"io"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inferencesurface"
	"overgo/internal/overgodb"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
)

// scorecardHeadings names each scorecard column the way a user reads it.
var scorecardHeadings = map[metric.Name]string{
	metric.FirstOutput: "First output",
	metric.Throughput:  "Throughput",
	metric.Quality:     "Held-out quality",
	metric.Failures:    "Failures",
	metric.Load:        "Load",
	metric.PeakMemory:  "Peak memory",
	metric.SetupSteps:  "Setup steps",
}

const (
	// bytesPerGiB renders device memory in the unit users read it in.
	bytesPerGiB = 1 << 30
	// percent renders a failure fraction as users read it.
	percent = 100
	// millisecondsPerSecond renders a wait of a second or more in seconds.
	millisecondsPerSecond = 1000
)

// writeScorecard prints the scorecard as a table, one row per activated
// model and task, then, when a served model declares an architecture, the
// share of them a declaration alone supports, then every gap with the reason
// no record measures it.
func writeScorecard(ctx context.Context, store *overgodb.Store, root string, output io.Writer) error {
	rows, err := scorecard.Build(ctx, store)
	if err != nil {
		return err
	}
	share := ""
	if slices.ContainsFunc(rows, func(row scorecard.Row) bool { return row.Architecture != "" }) {
		packages, err := inferencesurface.Packages(ctx, root)
		if err != nil {
			return err
		}
		named, err := servingNames(packages)
		if err != nil {
			return err
		}
		share = "\n" + declarationShare(rows, named)
	}
	var table strings.Builder
	table.WriteString("| Model | Task |")
	for _, column := range metric.Columns {
		table.WriteString(" " + scorecardHeadings[column] + " |")
	}
	table.WriteString("\n| --- | --- |" + strings.Repeat(" --- |", len(metric.Columns)) + "\n")
	var gaps []string
	for _, row := range rows {
		fmt.Fprintf(&table, "| %s | %s |", row.Name, row.Task)
		for _, column := range metric.Columns {
			cell := row.Cells[column]
			table.WriteString(" " + cellText(column, cell) + " |")
			if !cell.Measured() {
				gaps = append(gaps, fmt.Sprintf("- %s %s, %s: %s", row.Name, row.Task, scorecardHeadings[column], cell.Reason))
			}
		}
		table.WriteString("\n")
	}
	if _, err := io.WriteString(output, table.String()+share); err != nil {
		return err
	}
	if len(gaps) != 0 {
		_, err = io.WriteString(output, "\nGaps:\n"+strings.Join(gaps, "\n")+"\n")
	}
	return err
}

// tokenizerPackage keys on a model's declared tokenizer type and
// pre-tokenizer, whose names can coincide with an architecture's ("llama"
// names SentencePiece there): they select a vocabulary, not the architecture.
const tokenizerPackage = "/internal/tokenizer"

// servingNames maps each string literal in the serving packages' non-test Go
// files, outside the tokenizer, to the files that hold it: an architecture
// whose name is among them has serving code selected by that name.
func servingNames(packages []inferencesurface.Package) (map[string][]string, error) {
	named := map[string][]string{}
	for _, pkg := range packages {
		if strings.HasSuffix(pkg.ImportPath, tokenizerPackage) {
			continue
		}
		files, err := filepath.Glob(filepath.Join(pkg.Dir, "*.go"))
		if err != nil {
			return nil, err
		}
		for _, file := range files {
			if strings.HasSuffix(file, "_test.go") {
				continue
			}
			source, err := os.ReadFile(file)
			if err != nil {
				return nil, err
			}
			var scan scanner.Scanner
			scan.Init(token.NewFileSet().AddFile(file, -1, len(source)), source, nil, 0)
			name := path.Join(pkg.ImportPath, filepath.Base(file))
			for _, kind, literal := scan.Scan(); kind != token.EOF; _, kind, literal = scan.Scan() {
				if value, err := strconv.Unquote(literal); kind == token.STRING && err == nil && !slices.Contains(named[value], name) {
					named[value] = append(named[value], name)
				}
			}
		}
	}
	return named, nil
}

// declarationShare states how many served models a declaration alone
// supports: of the models whose activated recipe declares an architecture,
// those whose architecture name selects no serving code. Each architecture
// that does is listed with its models and the files that select it, the
// model-specific code a declaration would replace.
func declarationShare(rows []scorecard.Row, named map[string][]string) string {
	served := map[artifact.ID]bool{}
	models := map[string][]string{}
	alone := 0
	for _, row := range rows {
		if row.Architecture == "" || served[row.Model] {
			continue
		}
		served[row.Model] = true
		if len(named[row.Architecture]) == 0 {
			alone++
			continue
		}
		models[row.Architecture] = append(models[row.Architecture], row.Name)
	}
	var text strings.Builder
	text.WriteString(fmt.Sprintf("Supported by declaration alone: %d of %d served models; the architecture of each other selects serving code by name:\n", alone, len(served)))
	for _, architecture := range slices.Sorted(maps.Keys(models)) {
		text.WriteString(fmt.Sprintf("- %s (%s): %s\n", architecture, strings.Join(models[architecture], ", "), strings.Join(named[architecture], ", ")))
	}
	return text.String()
}

// cellText renders a measured cell with its unit and what it measures, and
// a gap as a dash; the gap's reason prints below the table.
func cellText(column metric.Name, cell scorecard.Cell) string {
	switch {
	case !cell.Measured():
		return "—"
	case column == metric.Failures:
		return fmt.Sprintf("%.3g%% (%s)", cell.Value*percent, cell.Label)
	case cell.Unit == "bytes":
		return fmt.Sprintf("%.3g GiB", cell.Value/bytesPerGiB)
	case cell.Unit == "ms" && cell.Value >= millisecondsPerSecond:
		return fmt.Sprintf("%.3g s", cell.Value/millisecondsPerSecond)
	case column == metric.Quality:
		return fmt.Sprintf("%.4g (%s)", cell.Value, cell.Label)
	case cell.Unit == "":
		return fmt.Sprintf("%.4g", cell.Value)
	default:
		return fmt.Sprintf("%.4g %s", cell.Value, cell.Unit)
	}
}
