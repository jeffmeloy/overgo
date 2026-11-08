package loop

import (
	"maps"
	"path"
	"slices"
)

// ReachedCells are the cells a change can move, of cells given by key with the directories of the checkout their
// measured path reads: those holding a changed path, each named once and in order. The owner: measure only what a
// change can move, not every model each time; a change no cell reads reaches none, and one model family's loader
// reaches that family's cells alone.
func ReachedCells(cells map[string][]string, changed []string) []string {
	var reached []string
	for _, key := range slices.Sorted(maps.Keys(cells)) {
		if slices.ContainsFunc(changed, func(file string) bool { return slices.Contains(cells[key], path.Dir(file)) }) {
			reached = append(reached, key)
		}
	}
	return reached
}
