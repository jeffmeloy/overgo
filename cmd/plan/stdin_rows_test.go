package main

import (
	"bytes"
	"encoding/json"
	"io/fs"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestRowsAreFiledFromStandardInput holds plan -add -from - to filing a plan-shaped document read from standard input
// as one read from a file is filed, before the row named, and to writing no file for it: the checkout holds the same
// files after as before, the plan among them. A document that is not one is refused and files nothing.
func TestRowsAreFiledFromStandardInput(t *testing.T) {
	t.Parallel()
	held := plan.Plan{Campaign: "held campaign", Items: []plan.Item{{ID: "earlier", Title: "Earlier", Status: plan.StatusOpen, Steps: []plan.Step{openStep("do")}}}}
	piped := plan.Plan{Items: []plan.Item{{ID: "piped", Title: "Piped", Owner: "master-lead", Status: plan.StatusOpen,
		Budget: plan.Budget{Nodes: 20, Reason: "It adds a listing."}, Steps: []plan.Step{openStep("do")}}}}
	root, written := editFixture(t, held)
	files := func() []string {
		var listed []string
		testcheck.NoError(t, filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
			// The system's own state, the plan's lock among it, is no draft.
			if entry != nil && entry.IsDir() && entry.Name() == processlock.StateDirectory {
				return filepath.SkipDir
			}
			listed = append(listed, path)
			return err
		}))
		return listed
	}
	before := files()
	document := testcheck.Must(json.Marshal(piped))(t)
	if output, err := planCommandReading(t, root, bytes.NewReader([]byte("not a plan")), "-add", "-from", standardInput); err == nil {
		t.Fatalf("plan -add -from - filed a document that is no plan:\n%s", output)
	}
	if !bytes.Equal(planBytes(t, filepath.Join(root, filepath.FromSlash(plan.Path))), written) {
		t.Fatal("a refused document changed the plan")
	}
	output, err := planCommandReading(t, root, bytes.NewReader(document), "-add", "-from", standardInput, "-before", "earlier")
	if err != nil || !strings.HasPrefix(output, "filed the rows and decisions of "+standardInputName+"; next: ") {
		t.Fatalf("plan -add -from -: %v\n%s", err, output)
	}
	if got, want := rowIDs(editedPlan(t, root)), "piped earlier"; got != want {
		t.Fatalf("rows after filing from standard input = %q, want %q", got, want)
	}
	if after := files(); !slices.Equal(after, before) {
		t.Fatalf("filing from standard input wrote files: %v, before %v", after, before)
	}
	requireNoCheckRan(t, root)
}
