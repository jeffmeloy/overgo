package landing

import (
	"slices"
	"strings"
)

// Process is one process that runs on this machine, by its image name, its
// id and the id of the process that started it.
type Process struct {
	Name       string
	ID, Parent int
}

// goCommand and testBinary are how a go command and a test binary are named,
// in lower case and without executableSuffix. A test binary's name ends with
// testBinary.
const goCommand, testBinary, executableSuffix = "go", ".test", ".exe"

// buildTools are the programs of the go toolchain that a build, a vet or a
// test run starts, named as goCommand is.
var buildTools = []string{"compile", "link", "asm", "cgo", "vet", "cover"}

// competing names, sorted and each once, the processes of a list that build
// or test beside the process self. A build tool competes, and so does a test
// binary. A go command competes only while it has no child, since one that
// waits on a child does no work of its own and the child is read by its own
// name. The ancestors of self are its own run and never compete, and neither
// does anything self started.
func competing(listed []Process, self int) []string {
	parent, waits := map[int]int{}, map[int]bool{}
	for _, process := range listed {
		parent[process.ID], waits[process.Parent] = process.Parent, true
	}
	// above lists a process's ancestors, nearest first, as far as the list
	// names them. An id that is its own parent, as the idle process is, and an
	// id met twice end the walk.
	above := func(id int) (chain []int) {
		for at, known := parent[id]; known && at != id && !slices.Contains(chain, at); at, known = parent[at] {
			chain = append(chain, at)
		}
		return chain
	}
	own := append(above(self), self)
	var names []string
	for _, process := range listed {
		name := strings.TrimSuffix(strings.ToLower(process.Name), executableSuffix)
		works := slices.Contains(buildTools, name) || strings.HasSuffix(name, testBinary) || name == goCommand && !waits[process.ID]
		if works && !slices.Contains(own, process.ID) && !slices.Contains(above(process.ID), self) {
			names = append(names, process.Name)
		}
	}
	slices.Sort(names)
	return slices.Compact(names)
}
