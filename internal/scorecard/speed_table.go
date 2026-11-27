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
// any warm speed was read for: its prompt processing (pp) at about each of evaluation.PromptLengths, the tokens it read
// uncached over the time to its first token (Cell.Rate), with how many it read (Cell.Tokens), since a reading counts
// at a length within 25% of it (SQA-336), and that time; and its token generation (tg), output tokens over decode time
// on a request again on the loaded model. A reading older than the record before carries its date (Record.Carry). A
// cell is read only on its warm contract, so a record measured before speed was read warm, or before its readings kept
// their token counts, gives no figure, and the first request after a load gives none.
func (record Record) SpeedTable() string {
	var table strings.Builder
	table.WriteString("| Model |")
	for _, length := range evaluation.PromptLengths {
		fmt.Fprintf(&table, " pp, about %s tokens |", thousands(length))
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
			cell := row.Lengths[at]
			seconds := time.Duration(cell.Value * float64(time.Millisecond)).Seconds()
			text := fmt.Sprintf("%s tok/s (%.2f s)", thousands(int(math.Round(cell.Rate))), seconds)
			if cell.Tokens > 0 {
				text = fmt.Sprintf("%s tok/s, %s tokens in %.2f s", thousands(int(math.Round(cell.Rate))), thousands(int(math.Round(cell.Tokens))), seconds)
			}
			cells = append(cells, text)
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
// for each image or video row with a time that names the preset it ran, a dash standing for a time that names none, a
// row with neither left out (a row whose timing request named no preset had hidden its reading at a person's size,
// SQA-513): its time at the size a person asks for with that preset, first, since that is the time a person gets; then
// the time of the scorecard's timing request with its preset, which for a video model is a small request that times its
// load and its prompt more than a clip (Wan's 32 x 16 request took 7.1 s of which its clip 0.34 s, 2026-11-25); then
// where the newest attempt at a person's size went (Row.Stages), with that attempt's own time, since the row's time is
// the median of the newest attempts and its parts are one attempt's (SenseNova's 130.7 s median beside parts of 10.9 s
// and 120.0 s, of an attempt of 132.3 s, SQA-512). A record measured before its attempts kept their presets gives no
// table.
func (record Record) GenerationTable() string {
	timed := func(cell Cell) string {
		return fmt.Sprintf("%.1f s (%s)", time.Duration(cell.Value*float64(time.Millisecond)).Seconds(), cell.Preset)
	}
	var lines strings.Builder
	for _, row := range record.Rows {
		if row.Task != recipe.TaskImageGen && row.Task != recipe.TaskVideoGen || row.Stale != "" {
			continue
		}
		small, sized, went := unreadSpeed, unreadSpeed, unreadSpeed
		if first := row.Cells[metric.FirstOutput]; first.Preset.Ran() {
			small = timed(first)
		}
		if at := slices.IndexFunc(row.Warm, func(cell Cell) bool { return cell.Contract.Metric == userSizeTimeMetric && cell.Preset.Ran() }); at >= 0 {
			sized, went = timed(row.Warm[at]), stagesText(row.Stages)
		}
		if small == unreadSpeed && sized == unreadSpeed {
			continue
		}
		fmt.Fprintf(&lines, "| %s | %s | %s | %s |\n", row.Name, sized, small, went)
	}
	if lines.Len() == 0 {
		return ""
	}
	return "\n| Model | time at the size a person asks for | time of the scorecard's timing request | where its newest attempt at a person's size went |\n| --- | --- | --- | --- |\n" + lines.String()
}

// stageShare is the share of a request's wall, one part in stageShare, a part takes at least to be named where its
// time went.
const stageShare = 20

// stagesText is where an attempt's wall went: the attempt's own time, its parts' sum, then its parts of a stageShare-th
// of it or more, in the order they ended.
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
	return fmt.Sprintf("%.1f s: %s", wall.Seconds(), strings.Join(named, ", "))
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
