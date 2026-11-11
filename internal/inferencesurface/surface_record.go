package inferencesurface

import (
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
)

// SurfaceRecord is the surface-move record, beneath the repository's root: what each move of the inference surface
// claims and the sets of names the identity code declares, which its set lines repeat.
var SurfaceRecord = filepath.Join("docs", "verification", "surface-moves.txt")

// recordSets are the record's set lines, each the set the code declares, by the key the record states it under.
func recordSets() map[string][]string {
	return map[string][]string{
		"set-surface-roots": surfaceRoots, "set-surface-kernels": surfaceKernels, "set-identity-addressed": identityAddressed,
		"set-build-settings": buildSettings, "set-result-settings": resultSettings, "set-libraries": libraries,
		"set-evaluation-binders": evaluationBinders, "set-capability-catalog": {capabilityCatalog},
		"set-capability-executor": {capabilityExecutor}, "set-text-runner": {textRunner},
	}
}

// WriteSurfaceRecord writes each set line of the record at path as the code declares the set, and leaves its other
// lines as they are, so the record's copy of a set is the tool's, never kept by hand (fail-early/record-written): a name
// added to the code (OVERGO_DEVICE_BUDGET_GIB, 2026-11-10) failed the record's test in the lanes until the line was
// typed. It writes nothing where every line is already the code's.
func WriteSurfaceRecord(path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	sets := recordSets()
	var written strings.Builder
	for line := range strings.Lines(string(raw)) {
		body := strings.TrimRight(line, "\r\n")
		if key, _, found := strings.Cut(body, ": "); found {
			if set, declared := sets[key]; declared {
				line = key + ": " + strings.Join(set, " ") + line[len(body):]
			}
		}
		written.WriteString(line)
	}
	if written.String() == string(raw) {
		return nil
	}
	return os.WriteFile(path, []byte(written.String()), clioptions.OutputFileMode)
}
