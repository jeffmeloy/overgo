package scorecard

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/scorecard/metric"
)

// unreadSpeed stands in the speed table where a row's warm speed was not read.
const unreadSpeed = "—"

// digitGroup is how many digits a thousands separator groups.
const digitGroup = 3

// speedOtherColumns are the speed table's columns beside its prompt lengths': the model's and tg's.
const speedOtherColumns = 2

// SpeedTable is the README's table of the record's warm speed, in the record's order, one line for each inference row
// any warm speed was read for: its prompt processing (pp) at each of evaluation.PromptLengths, the tokens it read
// uncached over the time to its first token (Cell.Rate) with that time beside it, and its token generation (tg),
// output tokens over decode time on a request again on the loaded model. A cell is read only on its warm contract, so
// a record measured before speed was read warm, or before its readings kept their token counts, gives no figure, and
// the first request after a load gives none.
func (record Record) SpeedTable() string {
	var table strings.Builder
	table.WriteString("| Model |")
	for _, length := range evaluation.PromptLengths {
		fmt.Fprintf(&table, " pp, %s-token prompt |", thousands(length))
	}
	table.WriteString(" tg |\n|" + strings.Repeat(" --- |", len(evaluation.PromptLengths)+speedOtherColumns) + "\n")
	for _, row := range record.Rows {
		if row.Task != recipe.TaskInference || row.Stale != "" {
			continue
		}
		var cells []string
		read := false
		for _, length := range evaluation.PromptLengths {
			at := slices.IndexFunc(row.Lengths, func(cell Cell) bool {
				return cell.Prompt == length && cell.Contract.Metric == lengthFirstMetric(length) && cell.Value > 0 && cell.Rate > 0
			})
			if at < 0 {
				cells = append(cells, unreadSpeed)
				continue
			}
			seconds := time.Duration(row.Lengths[at].Value * float64(time.Millisecond)).Seconds()
			cells = append(cells, fmt.Sprintf("%s tok/s (%.2f s)", thousands(int(math.Round(row.Lengths[at].Rate))), seconds))
			read = true
		}
		if rate := row.Cells[metric.Throughput]; rate.Contract.Metric == warmRateMetric && rate.Value > 0 {
			cells = append(cells, thousands(int(math.Round(rate.Value)))+" tok/s")
			read = true
		} else {
			cells = append(cells, unreadSpeed)
		}
		if read {
			fmt.Fprintf(&table, "| %s | %s |\n", row.Name, strings.Join(cells, " | "))
		}
	}
	return table.String()
}

// GenerationTable is the README's table of the record's generated pictures and clips, after its speed table: one line
// for each image or video row whose time to the whole output names the preset it ran, that time with its preset, and
// its time at the size a person asks for with that preset where one was read. A record measured before its attempts
// kept their presets gives no table.
func (record Record) GenerationTable() string {
	timed := func(cell Cell) string {
		return fmt.Sprintf("%.1f s (%s)", time.Duration(cell.Value*float64(time.Millisecond)).Seconds(), cell.Preset)
	}
	var lines strings.Builder
	for _, row := range record.Rows {
		first := row.Cells[metric.FirstOutput]
		if row.Task != recipe.TaskImageGen && row.Task != recipe.TaskVideoGen || row.Stale != "" || !first.Preset.Ran() {
			continue
		}
		sized := unreadSpeed
		if at := slices.IndexFunc(row.Warm, func(cell Cell) bool { return cell.Contract.Metric == userSizeTimeMetric && cell.Preset.Ran() }); at >= 0 {
			sized = timed(row.Warm[at])
		}
		fmt.Fprintf(&lines, "| %s | %s | %s |\n", row.Name, timed(first), sized)
	}
	if lines.Len() == 0 {
		return ""
	}
	return "\n| Model | time to the whole output | at the size a person asks for |\n| --- | --- | --- |\n" + lines.String()
}

// thousands writes a count with a comma between each group of three digits.
func thousands(count int) string {
	digits := strconv.Itoa(count)
	var grouped strings.Builder
	for index, digit := range digits {
		if index != 0 && (len(digits)-index)%digitGroup == 0 {
			grouped.WriteByte(',')
		}
		grouped.WriteRune(digit)
	}
	return grouped.String()
}
