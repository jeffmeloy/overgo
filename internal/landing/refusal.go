package landing

import (
	"cmp"
	"errors"
	"fmt"
	"io/fs"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"unicode/utf8"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
	"overgo/internal/testevidence"
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
	// Cause is the first line of the refusal's text past the gate's prefix, cut to causeBytes: what each plan review
	// counts refusals by, to find the brittle ones, where the class names only the stage (refusal-causes, 2026-11-11).
	Cause string `json:"cause,omitzero"`
}

// causeBytes bounds a refusal's cause in the record, which keeps every refusal: its first line says why.
const causeBytes = 240

// stageRules are the rules the rules stage holds a change to, each by its
// name and a phrase its refusal carries, which no other rule's does.
var stageRules = []struct{ name, phrase string }{
	{"pins", "changes bytes a record pins"},
	{"style", "new Go style findings"},
	{"legacy-forms", "new sort.Slice"},
	{"legacy-forms", "new sync.Once"},
	{"count", "production syntax nodes, and the consolidation"},
}

// pinsRefusal and shrinkRefusal are what the rules stage refuses a change with
// for moving bytes a record pins and a consolidation that grows past its count;
// each ends with the edit that clears it, a command where one does.
func pinsRefusal(stale []string) string {
	return "changes bytes a record pins: " + strings.Join(stale, "; ")
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
	cause, rest, _ := strings.Cut(cmp.Or(stage, refusal), "\n")
	if cause = saidFirst(strings.TrimSpace(cause), rest); len(cause) > causeBytes {
		cause = strings.ToValidUTF8(cause[:causeBytes], "")
	}
	classified.Cause = cause
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

// editSaid opens the edit a refusal says, which leads its cause.
const editSaid = "edit: "

// saidFirst is a refusal's first line led by the edit its text says (single-source/edits-said): the edit a rule names
// after its finding, or, for failed tests, the first failing test's last assertion, which says what it found and where;
// a line that says none stands as it is. The record cuts a cause to causeBytes, and of 25 refused landings listed on
// 2026-11-14 none kept the edit that clears it: a failed test's line named the tests alone, and a rule's edit came
// after a finding the cut ended.
func saidFirst(first, rest string) string {
	if before, edit, found := strings.Cut(first, editSaid); found {
		return editSaid + strings.TrimSuffix(strings.TrimSpace(edit), ")") + "; " + strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(before), "("))
	}
	if !strings.Contains(first, testevidence.FailedTests) {
		return first
	}
	var assertion string
	for line := range strings.Lines(rest) {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "--- FAIL") {
			break
		}
		if file, _, found := strings.Cut(line, ": "); found && strings.Contains(file, "_test.go:") {
			assertion = line
		}
	}
	if assertion == "" {
		return first
	}
	return editSaid + assertion + "; " + first
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

// WorkItem is a rule or test on the worklist: the refusals it made, how many of them said the edit that clears them,
// which the tool could make instead, and the latest one's cause.
type WorkItem struct {
	By       string `json:"by"`
	Refusals int    `json:"refusals"`
	Edits    int    `json:"edits,omitzero"`
	Latest   string `json:"latest"`
}

// Worklist lists the refusals the record at root keeps a cause for by what made each, most first: a refusal of the
// rules stage by its rules, one of failed tests by each test, and any other by its stage. Of each the question is what
// its refusals prevented: if nothing, it goes; otherwise its fix moves into the fix stage (the owner's review,
// 2026-11-12). The record kept no cause before session-leftovers/ended-session, so those refusals are not listed.
func Worklist(root string) ([]WorkItem, error) {
	_, refusals, err := stateRecords[Refusal](root, refusalRecord)
	if err != nil {
		return nil, err
	}
	var list []WorkItem
	for _, refusal := range refusals {
		if refusal.Cause == "" {
			continue
		}
		for _, by := range refusedBy(refusal) {
			at := slices.IndexFunc(list, func(item WorkItem) bool { return item.By == by })
			if at < 0 {
				list, at = append(list, WorkItem{By: by}), len(list)
			}
			list[at].Refusals++
			list[at].Latest = refusal.Cause
			if strings.HasPrefix(refusal.Cause, editSaid) {
				list[at].Edits++
			}
		}
	}
	slices.SortStableFunc(list, func(left, right WorkItem) int { return right.Refusals - left.Refusals })
	return list, nil
}

// refusedBy names what made a refusal: its rules, else each test it names failed, else its stage.
func refusedBy(refusal Refusal) []string {
	if len(refusal.Rules) > 0 {
		return refusal.Rules
	}
	_, failed, _ := strings.Cut(refusal.Cause, testevidence.FailedTests)
	names := strings.Split(failed, ", ")
	// A cause cut to causeBytes may end inside a name, which is left out.
	if len(refusal.Cause) > causeBytes-utf8.UTFMax {
		names = names[:len(names)-1]
	}
	var tests []string
	for _, name := range names {
		if strings.Contains(name, ": Test") {
			tests = append(tests, name)
		}
	}
	if len(tests) == 0 {
		return []string{refusal.Class}
	}
	return tests
}
