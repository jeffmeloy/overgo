package main

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/inferencesurface"
	"overgo/internal/jsonfile"
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

// A published record is stamped with the time it was published; its
// directory and record shape are the scorecard package's, which the
// workbench reads too.
const (
	scorecardRecordTime                = "20060102T150405Z"
	scorecardDirectoryMode fs.FileMode = 0o755
	scorecardRecordMode    fs.FileMode = 0o644
)

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
// no record measures it. With a records directory it then publishes the
// scorecard there and prints what moved since the previous record.
func writeScorecard(ctx context.Context, store *overgodb.Store, root, records string, output io.Writer) error {
	rows, err := scorecard.Build(ctx, store)
	if err != nil {
		return err
	}
	if err := printScorecard(ctx, rows, root, output); err != nil || records == "" {
		return err
	}
	commit, err := revParse(ctx, root, "HEAD")
	if err != nil {
		return err
	}
	name := scorecard.RecordName(time.Now().UTC().Format(scorecardRecordTime))
	moved, first, err := publishScorecard(records, name, commit, rows)
	if err != nil {
		return err
	}
	switch {
	case first:
		_, err = fmt.Fprintf(output, "\nPublished %s, the first record: nothing to compare with.\n", filepath.Join(records, name))
	case len(moved) == 0:
		_, err = fmt.Fprintf(output, "\nPublished %s: nothing moved since the previous record.\n", filepath.Join(records, name))
	default:
		_, err = fmt.Fprintf(output, "\nPublished %s. Since the previous record:\n%s\n", filepath.Join(records, name), strings.Join(moved, "\n"))
	}
	return err
}

// publishScorecard writes the rows as the directory's newest record and
// lists what moved since the previous one, as the scorecard package compares
// them. first reports that no previous record exists.
func publishScorecard(directory, name, commit string, rows []scorecard.Row) (moved []string, first bool, err error) {
	previous, err := scorecard.LatestRecords(directory, 1)
	if err != nil {
		return nil, false, err
	}
	current := scorecard.RecordOf(commit, rows)
	if err := os.MkdirAll(directory, scorecardDirectoryMode); err != nil {
		return nil, false, err
	}
	if err := jsonfile.Write(filepath.Join(directory, name), current, scorecardRecordMode); err != nil {
		return nil, false, err
	}
	if len(previous) == 0 {
		return nil, true, nil
	}
	return movedLines(scorecard.Moves(previous[0], current)), false, nil
}

// movedLines writes each move as one line of the publish report.
func movedLines(moves []scorecard.Move) []string {
	lines := make([]string, 0, len(moves))
	for _, move := range moves {
		row := move.Name + " " + move.Task
		heading := row + ", " + scorecardHeadings[move.Column]
		switch move.Kind {
		case scorecard.MoveAdded:
			lines = append(lines, "- "+row+": new")
		case scorecard.MoveRemoved:
			lines = append(lines, "- "+row+": gone")
		case scorecard.MoveMeasured:
			lines = append(lines, fmt.Sprintf("- %s: now measured, %.4g %s", heading, move.After, move.Unit))
		case scorecard.MoveUnmeasured:
			lines = append(lines, fmt.Sprintf("- %s: no longer measured: %s", heading, move.Reason))
		case scorecard.MoveChanged:
			lines = append(lines, fmt.Sprintf("- %s: %.4g to %.4g %s (%+.4g)", heading, move.Before, move.After, move.Unit, move.After-move.Before))
		}
	}
	return lines
}

// printScorecard writes the table, the declaration share and the gaps.
func printScorecard(ctx context.Context, rows []scorecard.Row, root string, output io.Writer) error {
	share := ""
	if slices.ContainsFunc(rows, func(row scorecard.Row) bool { return row.Architecture != "" }) {
		packages, err := inferencesurface.Packages(ctx, root)
		if err != nil {
			return err
		}
		named, err := inferencesurface.ServingNames(packages)
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
	var gaps, inapplicable []string
	for _, row := range rows {
		fmt.Fprintf(&table, "| %s | %s |", row.Name, row.Task)
		for _, column := range metric.Columns {
			cell := row.Cells[column]
			table.WriteString(" " + cellText(column, cell) + " |")
			line := fmt.Sprintf("- %s %s, %s: %s", row.Name, row.Task, scorecardHeadings[column], cell.Reason)
			switch {
			case cell.NotApplicable:
				inapplicable = append(inapplicable, line)
			case !cell.Measured():
				gaps = append(gaps, line)
			}
		}
		table.WriteString("\n")
	}
	text := table.String() + share
	if len(gaps) != 0 {
		text += "\nGaps:\n" + strings.Join(gaps, "\n") + "\n"
	}
	if len(inapplicable) != 0 {
		text += "\nNot applicable:\n" + strings.Join(inapplicable, "\n") + "\n"
	}
	_, err := io.WriteString(output, text)
	return err
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
	fmt.Fprintf(&text, "Supported by declaration alone: %d of %d served models", alone, len(served))
	if len(models) == 0 {
		return text.String() + ".\n"
	}
	text.WriteString("; the architecture of each other selects serving code by name:\n")
	for _, architecture := range slices.Sorted(maps.Keys(models)) {
		text.WriteString(fmt.Sprintf("- %s (%s): %s\n", architecture, strings.Join(models[architecture], ", "), strings.Join(named[architecture], ", ")))
	}
	return text.String()
}

// cellText renders a measured cell with its unit and what it measures, and
// a gap as a dash; the gap's reason prints below the table.
func cellText(column metric.Name, cell scorecard.Cell) string {
	switch {
	case cell.NotApplicable:
		return "n/a"
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
