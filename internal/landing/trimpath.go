package landing

import (
	"os"
	"slices"
	"strings"
)

// trimmedBuilds is the go flag every go command a landing, a run of the cycle's lanes and a measurement run starts
// builds with: go keys a build without it on the package's directory, so each tree a landing made fresh (the undo
// run's, the added-check run's) and each run in a new checkout built what it checks anew, and the build cache grew
// past 235 GB (gate-trimpath/do).
const trimmedBuilds = "-trimpath"

// TrimBuilds gives this process's go commands, and every process it starts, trimmed builds, keeping the flags GOFLAGS
// already holds; a go command that names -trimpath=false itself (go fix, which rewrites files by their paths) keeps
// its own.
func TrimBuilds() error {
	flags := os.Getenv("GOFLAGS")
	if slices.Contains(strings.Fields(flags), trimmedBuilds) {
		return nil
	}
	return os.Setenv("GOFLAGS", strings.TrimSpace(flags+" "+trimmedBuilds))
}
