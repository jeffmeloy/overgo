package main

import (
	"context"
	"fmt"
	"io"
	"strings"

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
// model and task, then every gap with the reason no record measures it.
func writeScorecard(ctx context.Context, store *overgodb.Store, output io.Writer) error {
	rows, err := scorecard.Build(ctx, store)
	if err != nil {
		return err
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
	if _, err := io.WriteString(output, table.String()); err != nil {
		return err
	}
	if len(gaps) != 0 {
		_, err = io.WriteString(output, "\nGaps:\n"+strings.Join(gaps, "\n")+"\n")
	}
	return err
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
