package scorecard

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/evaluation"
	"overgo/internal/processmeasure"
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
// its time at the size a person asks for with that preset where one was read, and where that time went (Row.Stages).
// A record measured before its attempts kept their presets gives no table.
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
		sized, went := unreadSpeed, unreadSpeed
		if at := slices.IndexFunc(row.Warm, func(cell Cell) bool { return cell.Contract.Metric == userSizeTimeMetric && cell.Preset.Ran() }); at >= 0 {
			sized, went = timed(row.Warm[at]), stagesText(row.Stages)
		}
		fmt.Fprintf(&lines, "| %s | %s | %s | %s |\n", row.Name, timed(first), sized, went)
	}
	if lines.Len() == 0 {
		return ""
	}
	return "\n| Model | time to the whole output | at the size a person asks for | where it went |\n| --- | --- | --- | --- |\n" + lines.String()
}

// stageShare is the share of a request's wall, one part in stageShare, a part takes at least to be named where its
// time went.
const stageShare = 20

// stagesText is where a request's wall went: its parts of a stageShare-th of it or more, in the order they ended.
func stagesText(stages []processmeasure.Part) string {
	var wall time.Duration
	for _, part := range stages {
		wall += part.Wall
	}
	var named []string
	for _, part := range stages {
		if wall > 0 && part.Wall*stageShare >= wall {
			named = append(named, fmt.Sprintf("%s %.1f s", part.Name, part.Wall.Seconds()))
		}
	}
	if len(named) == 0 {
		return unreadSpeed
	}
	return strings.Join(named, ", ")
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
