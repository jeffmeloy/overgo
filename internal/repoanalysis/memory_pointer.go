package repoanalysis

import (
	"maps"
	"slices"
	"strings"
)

// memoryIndex is the file of a project's session memory that the harness loads into every session, which holds one
// line pointing at the written guidance and nothing beside it.
const memoryIndex = "MEMORY.md"

// MemoryBeyondPointer names what a project's session memory holds beyond the one line of its index that points at the
// written guidance, of texts keyed as memory/<file>: each other file, and each line of the index after its first.
// Every session is told to write memory by itself, so memory cut back grows again, and rules written there disagree
// with skill.md; the session-start hook says what this finds, for it to be sorted into skill.md or code.
func MemoryBeyondPointer(texts map[string]string) []string {
	var beyond []string
	for _, name := range slices.Sorted(maps.Keys(texts)) {
		if !strings.HasSuffix(name, "/"+memoryIndex) {
			beyond = append(beyond, name)
			continue
		}
		var lines []string
		for line := range strings.Lines(texts[name]) {
			if line = strings.TrimSpace(line); line != "" {
				lines = append(lines, line)
			}
		}
		for _, line := range lines[min(1, len(lines)):] {
			beyond = append(beyond, name+": "+line)
		}
	}
	return beyond
}
