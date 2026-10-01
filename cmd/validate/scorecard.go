package main

import (
	"context"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"
	"time"

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
	previous, err := scorecard.Records(records)
	if err != nil {
		return err
	}
	current := scorecard.Record{Commit: commit, Rows: rows}
	path, err := scorecard.Publish(records, time.Now(), current)
	if err != nil {
		return err
	}
	if len(previous) == 0 {
		_, err = fmt.Fprintf(output, "\nPublished %s, the first record: nothing to compare with.\n", path)
		return err
	}
	if moved := scorecardMoves(previous[len(previous)-1], current); len(moved) != 0 {
		_, err = fmt.Fprintf(output, "\nPublished %s. Since the previous record:\n%s\n", path, strings.Join(moved, "\n"))
		return err
	}
	_, err = fmt.Fprintf(output, "\nPublished %s: nothing moved since the previous record.\n", path)
	return err
}

// promotionBaseline is the scorecard the cycle's next record is judged
// against: of those published beside the store, the newest that measured the
// commit the promoted branch holds, as Baseline chooses it for the record the
// cycle publishes after them. Nil when none did.
func promotionBaseline(ctx context.Context, root, storeRoot string) (*scorecard.Record, error) {
	published, err := scorecard.Records(scorecard.Directory(storeRoot))
	if err != nil {
		return nil, err
	}
	return scorecard.Baseline(ctx, root, append(published, scorecard.Record{}))
}

// openGap fails while the gap key names is open on the scorecard the store
// holds now, judged against the newest published record's baseline: the
// verifier of the row the loop proposed for the gap.
func openGap(ctx context.Context, store *overgodb.Store, root, records, key string, output io.Writer) error {
	rows, err := scorecard.Build(ctx, store)
	if err != nil {
		return err
	}
	published, err := scorecard.Records(records)
	if err != nil {
		return err
	}
	baseline, err := scorecard.Baseline(ctx, root, published)
	if err != nil {
		return err
	}
	gaps := scorecard.Record{Rows: rows}.Gaps(baseline)
	if at := slices.IndexFunc(gaps, func(gap scorecard.Gap) bool { return gap.Key == key }); at >= 0 {
		return fmt.Errorf("%s is open:\n%s", key, gaps[at].Detail)
	}
	_, err = fmt.Fprintf(output, "%s is closed.\n", key)
	return err
}

// scorecardMoves lists what moved from the previous record to the current,
// by the scorecard's comparison of rows of one model and task: each cell
// measured on the same contract whose value changed, each cell that gained
// its measure, each the comparison cannot judge and why, and each row that
// is new or gone.
func scorecardMoves(previous, current scorecard.Record) []string {
	var moved []string
	for _, pair := range current.Pairs(previous) {
		switch {
		case pair.Before == nil:
			moved = append(moved, "- "+pair.After.Name+" "+string(pair.After.Task)+": new")
			continue
		case pair.After == nil:
			moved = append(moved, "- "+pair.Before.Name+" "+string(pair.Before.Task)+": gone")
			continue
		}
		for _, column := range metric.Columns {
			now, then := pair.After.Cells[column], pair.Before.Cells[column]
			heading := pair.After.Name + " " + string(pair.After.Task) + ", " + scorecardHeadings[column]
			switch verdict, why := pair.Compare(column); {
			case verdict == scorecard.Unknown:
				moved = append(moved, "- "+heading+": "+why)
			case now.Measured() && !then.Measured():
				moved = append(moved, fmt.Sprintf("- %s: now measured, %.4g %s", heading, now.Value, now.Unit))
			case now.Value != then.Value:
				moved = append(moved, fmt.Sprintf("- %s: %.4g to %.4g %s (%+.4g)", heading, then.Value, now.Value, now.Unit, now.Value-then.Value))
			}
		}
	}
	return moved
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
