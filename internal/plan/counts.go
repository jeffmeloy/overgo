package plan

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// LandedCount is what one landing counted, as its message states it: the production syntax nodes its change added, and
// the paths it changed.
type LandedCount struct {
	Nodes int
	Paths []string
}

const (
	// landedFiles opens the line of a landing's message that names the paths it changed.
	landedFiles = "- files: "
	// landedNodes opens what a landing's rules stage counted of the production code it added, before " nodes".
	landedNodes = "(production "
)

// ReadLandedCount reads a landing's message: the paths its files line names and the nodes its rules stage counted. A
// message that states either not is no landing's count.
func ReadLandedCount(message string) (LandedCount, bool) {
	var count LandedCount
	read := false
	for line := range strings.Lines(message) {
		if listed, found := strings.CutPrefix(strings.TrimSpace(line), landedFiles); found {
			count.Paths = strings.Split(listed, ", ")
		}
		if _, after, found := strings.Cut(line, landedNodes); found {
			number, _, _ := strings.Cut(after, " ")
			nodes, err := strconv.Atoi(number)
			count.Nodes, read = nodes, err == nil
		}
	}
	return count, read && len(count.Paths) > 0
}

// HistoryCount is the count a new step that lists files is filed with: the median of what the landings that changed a
// path the files list counted, and how many they were. No landing that did gives none. Counts filed by hand ran far
// under what landed (wan-few-step 300 -> 2950, umt5-resident 336 -> 1750, sampling-split 400 -> 1700), and on
// 2026-11-08 the lead raised all 24 media rows about threefold by hand.
func HistoryCount(history []LandedCount, files []string) (nodes, landings int) {
	var counted []int
	for _, landed := range history {
		if slices.ContainsFunc(landed.Paths, func(path string) bool { return Lists(files, path) }) {
			counted = append(counted, landed.Nodes)
		}
	}
	if len(counted) == 0 {
		return 0, 0
	}
	slices.Sort(counted)
	// The median is the middle of the sorted counts.
	return counted[len(counted)>>1], len(counted)
}

// HistoryReason says where a count HistoryCount gave came from, with the figure a session wrote beside it, if any.
func HistoryReason(nodes, landings, figure int, files []string) string {
	reason := fmt.Sprintf("%d, the median of what %d landings that changed %s counted", nodes, landings, strings.Join(files, ", "))
	if landings == 0 {
		reason = "no landing changed " + strings.Join(files, ", ")
	}
	if figure != 0 {
		reason += fmt.Sprintf("; the session's figure %d", figure)
	}
	return reason
}
