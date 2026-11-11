package landing

import (
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"slices"
	"strings"
	"syscall"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
)

// Refusal classes a refused landing by the stage that refused it, the gate
// when none did, or a cause outside the change that the owner controls and
// the probe that passes once it has cleared.
type Refusal struct {
	Reference string `json:"reference"`
	Class     string `json:"class"`
	External  string `json:"external,omitzero"`
	Probe     string `json:"probe,omitzero"`
	// Recurrences counts the earlier refusals of the class: of the row, or
	// of any row for an external cause.
	Recurrences int `json:"recurrences,omitzero"`
	// Rules names the rules a refusal of the rules stage broke, so a rule
	// whose refusals are not defects can be counted, and found.
	Rules []string `json:"rules,omitzero"`
}

// stageRules are the rules the rules stage holds a change to, each by its
// name and a phrase its refusal carries, which no other rule's does.
var stageRules = []struct{ name, phrase string }{
	{"pins", "changes bytes a record pins"},
	{"style", "new Go style findings"},
	{"legacy-forms", "new sort.Slice"},
	{"legacy-forms", "new sync.Once"},
	{"cuda-lanes", "has CUDA-gated tests that no lane runs"},
	{"count", "production syntax nodes, and the consolidation"},
}

// pinsRefusal, uncoveredRefusal and shrinkRefusal are what the rules stage
// refuses a change with for moving bytes a record pins, a CUDA-gated test no
// lane runs, and a consolidation that grows past its count; each ends with
// the edit that clears it, a command where one does.
func pinsRefusal(stale []string) string {
	return "changes bytes a record pins: " + strings.Join(stale, "; ")
}

func uncoveredRefusal(directory string) string {
	return directory + " has CUDA-gated tests that no lane runs; edit: add \"./" + directory + "\" to cudaPackages in internal/landing/lanes.go"
}

func shrinkRefusal(growth int, row string, left int) string {
	return fmt.Sprintf("adds %d production syntax nodes, and the consolidation %s may add %d; edit: remove production code until it adds no more than that, or file the growth as a row of its own", growth, row, left)
}

const (
	// gateClass is a refusal no stage made: the gate did not build, or it
	// refused before or after its checks.
	gateClass = "gate"
	// quarantineClass is a built binary the host's antivirus removed.
	quarantineClass = "quarantine"
	// quarantined is ERROR_VIRUS_INFECTED, whose text a build prints when
	// Windows quarantined its output.
	quarantined syscall.Errno = 225
	// refusalRecord holds the refusals in the checkout's process state.
	refusalRecord = "landing_refusals.json"
)

// Classify classes a refusal from the text the gate, or its build, reported.
func Classify(reference, refusal string) Refusal {
	if strings.Contains(refusal, quarantined.Error()) {
		return Refusal{Reference: reference, Class: quarantineClass, Probe: "go build -o <scratch>/gate.exe ./cmd/gate",
			External: "Windows Defender quarantined a binary the landing built; an exclusion or a false-positive submission is the owner's action"}
	}
	classified := Refusal{Reference: reference, Class: gateClass}
	_, stage, _ := strings.Cut(refusal, stagePrefix)
	// The stage space is none of checks, and what it refuses is the machine, not the change.
	if strings.HasPrefix(stage, spaceClass+":") {
		classified.Class, classified.Probe = spaceClass, "go run ./cmd/gate -plan <item>/<step> -message-file - -preflight"
		classified.External = "the volume a commit's checks write to has less free space than their floor, or its free space could not be read; freeing it is the owner's action, and go run ./cmd/loop -leftovers lists what can be removed"
	}
	for _, check := range (&landing{}).checks() {
		if strings.HasPrefix(stage, check.name+":") {
			classified.Class = check.name
		}
	}
	if classified.Class == "rules" {
		for _, rule := range stageRules {
			if strings.Contains(stage, rule.phrase) && !slices.Contains(classified.Rules, rule.name) {
				classified.Rules = append(classified.Rules, rule.name)
			}
		}
	}
	return classified
}

// stateRecords reads the list the checkout at root keeps under name in its
// process state: where the list is kept, and the records it holds, none when
// it was never written.
func stateRecords[Record any](root, name string) (record string, earlier []Record, err error) {
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return "", nil, err
	}
	record = filepath.Join(root, processlock.StateDirectory, name)
	if err := jsonfile.Decode(record, &earlier); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return "", nil, err
	}
	return record, earlier, nil
}

// RecordRefusal adds the refusal, its recurrences counted, to the record of
// the checkout at root.
func RecordRefusal(root string, refusal Refusal) (Refusal, error) {
	record, earlier, err := stateRecords[Refusal](root, refusalRecord)
	if err != nil {
		return refusal, err
	}
	for _, past := range earlier {
		if past.Class == refusal.Class && (refusal.External != "" || past.Reference == refusal.Reference) {
			refusal.Recurrences++
		}
	}
	return refusal, jsonfile.Write(record, append(earlier, refusal), clioptions.PrivateFileMode)
}
