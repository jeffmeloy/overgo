package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestAStepsNotesAreShownWithIt holds a step's working notes to the plan and to the task's prompt: plan -note keeps
// each as a paragraph that opens "Note: ", once however it was written, and the prompt shows every note of the step,
// in order and each on one line, and none of the reasoning beside them.
func TestAStepsNotesAreShownWithIt(t *testing.T) {
	t.Parallel()
	step := openStep("do")
	step.Rationale = "Why the step exists, at length.\n\nServes: the loop."
	root, _ := editFixture(t, plan.Plan{Items: []plan.Item{{ID: "work", Title: "Work", Status: plan.StatusOpen, Steps: []plan.Step{step}}}})
	for index, words := range []string{"The 9B's 4K prompt read\n2.84 s cold.", notePrefix + "Read warm, 1.41 s."} {
		file := filepath.Join(t.TempDir(), "note.txt")
		testcheck.NoError(t, os.WriteFile(file, []byte(words), 0o600))
		if output, err := planCommand(t, root, "-note", "-rationale-file", file, "work", "do"); err != nil {
			t.Fatalf("note %d: %v\n%s", index, err, output)
		}
	}
	noted := editedPlan(t, root).Items[0]
	if strings.Count(noted.Steps[0].Rationale, notePrefix) != 2 {
		t.Fatalf("the step's reasoning after two notes:\n%s\nwant each note opened once by %q", noted.Steps[0].Rationale, notePrefix)
	}
	var prompt bytes.Buffer
	printPromptStep(noted, noted.Steps[0], nil, &prompt)
	shown := prompt.String()
	if !strings.Contains(shown, "NOTE The 9B's 4K prompt read 2.84 s cold.\nNOTE Read warm, 1.41 s.\n") || strings.Contains(shown, "Why the step exists") {
		t.Fatalf("the task's prompt =\n%s\nwant both notes, in order and each on one line, and none of the reasoning", shown)
	}
}
