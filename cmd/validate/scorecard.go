package main

import (
	"context"
	"fmt"
	"io"
	"strings"

	"overgo/internal/overgodb"
	"overgo/internal/scorecard"
)

// scorecardHeadings names each scorecard column the way a user reads it.
var scorecardHeadings = map[scorecard.Metric]string{
	scorecard.FirstOutput: "First output",
	scorecard.Throughput:  "Throughput",
	scorecard.Quality:     "Held-out quality",
	scorecard.Failures:    "Failures",
	scorecard.Load:        "Load",
	scorecard.PeakMemory:  "Peak memory",
	scorecard.SetupSteps:  "Setup steps",
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
	for _, metric := range scorecard.Metrics {
		table.WriteString(" " + scorecardHeadings[metric] + " |")
	}
	table.WriteString("\n| --- | --- |" + strings.Repeat(" --- |", len(scorecard.Metrics)) + "\n")
	var gaps []string
	for _, row := range rows {
		fmt.Fprintf(&table, "| %s | %s |", row.Name, row.Task)
		for _, metric := range scorecard.Metrics {
			cell := row.Cells[metric]
			table.WriteString(" " + cellText(metric, cell) + " |")
			if !cell.Measured() {
				gaps = append(gaps, fmt.Sprintf("- %s %s, %s: %s", row.Name, row.Task, scorecardHeadings[metric], cell.Reason))
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
func cellText(metric scorecard.Metric, cell scorecard.Cell) string {
	switch {
	case !cell.Measured():
		return "—"
	case metric == scorecard.Failures:
		return fmt.Sprintf("%.3g%% (%s)", cell.Value*percent, cell.Label)
	case cell.Unit == "bytes":
		return fmt.Sprintf("%.3g GiB", cell.Value/bytesPerGiB)
	case cell.Unit == "ms" && cell.Value >= millisecondsPerSecond:
		return fmt.Sprintf("%.3g s", cell.Value/millisecondsPerSecond)
	case metric == scorecard.Quality:
		return fmt.Sprintf("%.4g (%s)", cell.Value, cell.Label)
	case cell.Unit == "":
		return fmt.Sprintf("%.4g", cell.Value)
	default:
		return fmt.Sprintf("%.4g %s", cell.Value, cell.Unit)
	}
}
