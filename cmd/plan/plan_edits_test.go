package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/plan"
	"overgo/internal/worklease"
)

// planCommandEnvironment marks a child process a test started as the plan
// command.
const planCommandEnvironment = "OVERGO_PLAN_TEST_COMMAND"

// editCheck is the check the fixture steps carry. An edit runs no check, so
// each test asserts the file it would leave is absent.
const editCheck = "touch check-ran"

// TestMain runs the tests, or, in a child a test started as the plan command,
// the command: its own main over the arguments it was started with, so the
// flags a test gives are the flags the command registers and parses, in the
// order Go's flag parsing takes them.
func TestMain(m *testing.M) {
	if os.Getenv(planCommandEnvironment) == "" {
		os.Exit(m.Run())
	}
	// The command's flags alone: the test binary's own are not the command's.
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	main()
	// main returns only when the command succeeded.
	os.Exit(0)
}

// planCommand runs the plan command in root with the arguments a person
// would type, and returns what it printed and whether it refused.
func planCommand(t *testing.T, root string, arguments ...string) (string, error) {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(t.Context(), executable, arguments...)
	command.Dir = root
	// The child's refusals are this test's to read: they go to no error file a
	// caller of the tests named.
	command.Env = append(os.Environ(), planCommandEnvironment+"=1", plan.AutomationRoleEnvironment+"="+worklease.UnassignedRole, clioptions.ErrorFileEnvironment+"=")
	output, err := command.CombinedOutput()
	return string(output), err
}

// editFixture is a root holding document as its plan, with the plan's bytes.
func editFixture(t *testing.T, document plan.Plan) (root string, written []byte) {
	t.Helper()
	root = t.TempDir()
	return root, planBytes(t, saveMutationPlan(t, root, document))
}

// planBytes reads a plan file whole.
func planBytes(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// editedPlan loads the plan of root as the command left it.
func editedPlan(t *testing.T, root string) plan.Plan {
	t.Helper()
	document, err := plan.Load(filepath.Join(root, filepath.FromSlash(plan.Path)))
	if err != nil {
		t.Fatal(err)
	}
	return document
}

// rendered is a value as the plan file would hold it, for comparing rows and
// steps whole.
func rendered(t *testing.T, value any) string {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// rowIDs are the ids of a plan's rows in order.
func rowIDs(document plan.Plan) string {
	var ids []string
	for _, item := range document.Items {
		ids = append(ids, item.ID)
	}
	return strings.Join(ids, " ")
}

// stepIDs are the ids of a row's steps in order.
func stepIDs(item plan.Item) string {
	var ids []string
	for _, step := range item.Steps {
		ids = append(ids, step.ID)
	}
	return strings.Join(ids, " ")
}

// requireRefused fails unless the command refused, said why in the words
// given, left the plan's bytes as they were and ran no check.
func requireRefused(t *testing.T, root string, written []byte, why string, arguments ...string) {
	t.Helper()
	output, err := planCommand(t, root, arguments...)
	if err == nil || !strings.Contains(output, why) {
		t.Fatalf("plan %v = %v, want a refusal naming %q:\n%s", arguments, err, why, output)
	}
	if after := planBytes(t, filepath.Join(root, filepath.FromSlash(plan.Path))); !bytes.Equal(after, written) {
		t.Fatalf("the refused plan %v changed the plan:\n%s", arguments, after)
	}
	requireNoCheckRan(t, root)
}

// requireNoCheckRan fails if a step's check ran in root.
func requireNoCheckRan(t *testing.T, root string) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(root, "check-ran")); err == nil {
		t.Fatal("the edit ran a step's check")
	}
}

// requireOffered fails unless the command's help carries each text, and the
// refusal of a call that names no operation lists the flag.
func requireOffered(t *testing.T, root, name string, texts ...string) {
	t.Helper()
	help, err := planCommand(t, root, "-h")
	if err != nil {
		t.Fatalf("plan -h: %v\n%s", err, help)
	}
	for _, text := range texts {
		if !strings.Contains(help, text) {
			t.Fatalf("the help does not carry %q:\n%s", text, help)
		}
	}
	if refusal, err := planCommand(t, root); err == nil || !strings.Contains(refusal, " "+name+",") {
		t.Fatalf("a call with no operation = %v, want a refusal that lists %s:\n%s", err, name, refusal)
	}
}

// openStep is an open fixture step.
func openStep(id string) plan.Step {
	return plan.Step{ID: id, Title: "Step " + id, Status: plan.StatusOpen, Verify: editCheck}
}

// TestAddFilesEveryRowOfADocument holds plan -add -from to filing a
// plan-shaped document whole: every row arrives with its steps, count and
// reasoning, in the document's order, before the row named or at the top, and
// every decision joins the plan's. No row the plan held changes, the
// close-out row's list among them, and the document's own campaign and
// doctrine are not taken. A document naming a row or a decision the plan
// holds is refused and files nothing. Plain -add, which enters its one task
// through the same insertion, still makes the close-out row await it.
func TestAddFilesEveryRowOfADocument(t *testing.T) {
	t.Parallel()
	held := plan.Plan{
		Campaign: "held campaign", Doctrine: []plan.Paragraph{{Governs: "standing", Text: "held doctrine"}},
		Items: []plan.Item{
			{ID: "earlier", Title: "Earlier", Status: plan.StatusOpen, Steps: []plan.Step{openStep("do")}},
			{ID: "campaign-closeout", Title: "Close out", Status: plan.StatusOpen, Steps: []plan.Step{
				{ID: "do", Title: "Close out", Status: plan.StatusOpen, Verify: editCheck, DependsOn: []string{"earlier/do"}},
			}},
		},
		Decisions: []plan.Decision{{ID: "held-question", Question: "Which?", Options: []string{"a", "b"}, Recommendation: "a", Asked: "2026-10-01"}},
	}
	filed := plan.Plan{
		Campaign: "an amendment", Doctrine: []plan.Paragraph{{Governs: "standing", Text: "the amendment's own doctrine"}},
		Items: []plan.Item{
			{ID: "filed-first", Title: "First filed", Owner: "master-lead", Status: plan.StatusOpen,
				Budget: plan.Budget{Nodes: 40, Reason: "It adds a listing."},
				Steps: []plan.Step{
					{ID: "inventory", Title: "List", Status: plan.StatusOpen, Verify: editCheck, Rationale: "Why the listing comes first."},
					{ID: "cleanup", Title: "Clean", Status: "blocked: waits for the owner", Verify: editCheck, DependsOn: []string{"filed-first/inventory"}},
				}},
			{ID: "filed-second", Title: "Second filed", Class: plan.ClassConsolidation, Status: plan.StatusOpen, Steps: []plan.Step{openStep("do")}},
		},
		Decisions: []plan.Decision{{ID: "filed-question", Question: "Whether?", Options: []string{"yes", "no"}, Recommendation: "no", Blocks: []string{"filed-second"}, Asked: "2026-10-12"}},
	}
	save := func(name string, document plan.Plan) string {
		t.Helper()
		path := filepath.Join(t.TempDir(), name)
		if err := plan.Save(path, document); err != nil {
			t.Fatal(err)
		}
		return path
	}
	amendment := save("amendment.json", filed)

	root, _ := editFixture(t, held)
	if output, err := planCommand(t, root, "-add", "-from", amendment, "-before", "campaign-closeout"); err != nil || !strings.HasPrefix(output, "filed the rows and decisions of "+amendment+"; next: ") {
		t.Fatalf("plan -add -from: %v\n%s", err, output)
	}
	after := editedPlan(t, root)
	if got, want := rowIDs(after), "earlier filed-first filed-second campaign-closeout"; got != want {
		t.Fatalf("rows after filing before the close-out row = %q, want %q", got, want)
	}
	if rendered(t, after.Items[1:3]) != rendered(t, filed.Items) {
		t.Fatalf("the filed rows differ from the document's:\n%s\nwant\n%s", rendered(t, after.Items[1:3]), rendered(t, filed.Items))
	}
	if rendered(t, after.Items[0]) != rendered(t, held.Items[0]) || rendered(t, after.Items[3]) != rendered(t, held.Items[1]) {
		t.Fatalf("filing a document changed a row the plan held, the close-out row's list among them:\n%s", rendered(t, after.Items))
	}
	if rendered(t, after.Decisions) != rendered(t, append(held.Decisions, filed.Decisions...)) {
		t.Fatalf("decisions after filing = %s", rendered(t, after.Decisions))
	}
	if after.Campaign != held.Campaign || !reflect.DeepEqual(after.Doctrine, held.Doctrine) {
		t.Fatalf("filing took the document's campaign or doctrine: %q, %+v", after.Campaign, after.Doctrine)
	}
	requireNoCheckRan(t, root)

	top, _ := editFixture(t, held)
	if output, err := planCommand(t, top, "-add", "-from", amendment); err != nil {
		t.Fatalf("plan -add -from with no -before: %v\n%s", err, output)
	}
	if got, want := rowIDs(editedPlan(t, top)), "filed-first filed-second earlier campaign-closeout"; got != want {
		t.Fatalf("rows after filing with no -before = %q, want %q", got, want)
	}

	output, err := planCommand(t, top, "-add", "-title", "One task", "-vcmd", editCheck, "-role", "lane-a", "-before", "earlier", "one-task")
	if err != nil || !strings.HasPrefix(output, "added item one-task (step do); next: ") {
		t.Fatalf("plain plan -add: %v\n%s", err, output)
	}
	one := plan.Item{ID: "one-task", Title: "One task", Owner: "lane-a", Status: plan.StatusOpen, Steps: []plan.Step{{ID: "do", Title: "One task", Status: plan.StatusOpen, Verify: editCheck}}}
	awaiting := held.Items[1]
	awaiting.Steps = []plan.Step{awaiting.Steps[0]}
	awaiting.Steps[0].DependsOn = []string{"earlier/do", "one-task/do"}
	if added := editedPlan(t, top); rowIDs(added) != "filed-first filed-second one-task earlier campaign-closeout" ||
		rendered(t, added.Items[2]) != rendered(t, one) || rendered(t, added.Items[4]) != rendered(t, awaiting) {
		t.Fatalf("plan after a plain -add:\n%s", rendered(t, added.Items))
	}

	refusing, written := editFixture(t, held)
	heldRow := filed
	heldRow.Items = []plan.Item{filed.Items[1], held.Items[0]}
	requireRefused(t, refusing, written, `item "earlier" already exists`, "-add", "-from", save("held-row.json", heldRow))
	heldDecision := filed
	heldDecision.Decisions = held.Decisions
	requireRefused(t, refusing, written, `"held-question"`, "-add", "-from", save("held-decision.json", heldDecision))
	requireRefused(t, refusing, written, "absent.json", "-add", "-from", filepath.Join(refusing, "absent.json"))
	requireRefused(t, refusing, written, `"absent"`, "-add", "-from", amendment, "-before", "absent")
	requireRefused(t, refusing, written, "usage: plan -add -from <document> [-before <id>]", "-add", "-from", amendment, "stray-name")
	requireOffered(t, refusing, "-add", "-add -from <document> [-before <id>]", "an id the plan holds is refused, and no row the plan holds changes")
}

// TestSetCountWritesTheCountAndItsReason holds plan -setcount to writing what
// a row may still add and why, a count below zero included, with the reason
// given in words or in a file. It leaves the row's other fields and every
// other row as they were, and runs no check although the row's step is the
// one a dispatch would give. It refuses a count with no reason, a row the
// plan lacks and a count that names a step, since a count is a row's, and its
// flags come before the row's name, as Go's flag parsing requires.
func TestSetCountWritesTheCountAndItsReason(t *testing.T) {
	t.Parallel()
	held := plan.Plan{Items: []plan.Item{
		{ID: "grows", Title: "Grows", Status: plan.StatusOpen, Budget: plan.Budget{Nodes: 3, Paydown: "other", Reason: "The old reason."}, Steps: []plan.Step{openStep("do")}},
		{ID: "other", Title: "Other", Status: plan.StatusOpen, Budget: plan.Budget{Nodes: 7, Reason: "Its own reason."}, Steps: []plan.Step{openStep("do")}},
	}}
	root, _ := editFixture(t, held)
	output, err := planCommand(t, root, "-setcount", "-nodes", "120", "-reason", "The listing was counted at 120.", "grows")
	if err != nil || !strings.HasPrefix(output, "grows: count set to 120 nodes; next: ") {
		t.Fatalf("plan -setcount: %v\n%s", err, output)
	}
	after := editedPlan(t, root)
	want := held.Items[0]
	want.Budget = plan.Budget{Nodes: 120, Paydown: "other", Reason: "The listing was counted at 120."}
	if rendered(t, after.Items[0]) != rendered(t, want) || rendered(t, after.Items[1]) != rendered(t, held.Items[1]) {
		t.Fatalf("rows after setting the count of the first:\n%s", rendered(t, after.Items))
	}
	requireNoCheckRan(t, root)
	reason := filepath.Join(t.TempDir(), "reason.txt")
	if err := os.WriteFile(reason, []byte("\nThe row must shrink.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if output, err := planCommand(t, root, "-setcount", "-nodes", "-7", "-rationale-file", reason, "other"); err != nil {
		t.Fatalf("plan -setcount below zero, its reason in a file: %v\n%s", err, output)
	}
	if got := editedPlan(t, root).Items[1].Budget; got != (plan.Budget{Nodes: -7, Reason: "The row must shrink."}) {
		t.Fatalf("count below zero = %+v", got)
	}

	refusing, written := editFixture(t, held)
	const usage = "plan -setcount -nodes <n> -reason <why> <item-id>"
	requireRefused(t, refusing, written, usage, "-setcount", "-nodes", "120", "grows")
	requireRefused(t, refusing, written, usage, "-setcount", "-nodes", "120", "-reason", " ", "grows")
	requireRefused(t, refusing, written, usage, "-setcount", "-nodes", "120", "-reason", "A step has no count.", "grows", "do")
	requireRefused(t, refusing, written, usage, "-setcount", "-nodes", "120", "-reason", "A step has no count.", "grows/do")
	requireRefused(t, refusing, written, "usage, with every flag before the names, where Go's flag parsing stops:", "-setcount", "grows", "-nodes", "120", "-reason", "The flags follow the name.")
	requireRefused(t, refusing, written, `"absent"`, "-setcount", "-nodes", "120", "-reason", "No such row.", "absent")
	requireRefused(t, refusing, written, "not found", "-setcount", "-nodes", "120", "-reason", "No row named.")
	requireRefused(t, refusing, written, "absent.txt", "-setcount", "-nodes", "120", "-rationale-file", filepath.Join(refusing, "absent.txt"), "grows")
	requireOffered(t, refusing, "-setcount", "running no check: -setcount -nodes <n> -reason <why> <item-id>", "in place of -reason: a file that holds the edit's words",
		"0, the default: it may not grow; below zero: it must shrink", "with -setcount, -addstep, -note or -retire: the edit's words")
}

// TestAddStepInsertsBeforeAnother holds plan -addstep to putting an open step,
// with its title, check and reasoning, before the step named, or last when
// none is named, and leaving the row's other steps and every other row as
// they were. The reasoning comes from a file or from -reason, and the row and
// step are two words or one joined by a slash. It refuses a step the row
// holds, a -before that names no step, a row the plan lacks, and a step with
// no name, no title or a blank one, no check or no reasoning.
func TestAddStepInsertsBeforeAnother(t *testing.T) {
	t.Parallel()
	held := plan.Plan{Items: []plan.Item{
		{ID: "work", Title: "Work", Status: plan.StatusOpen, Steps: []plan.Step{openStep("first"), openStep("last")}},
		{ID: "other", Title: "Other", Status: plan.StatusOpen, Steps: []plan.Step{openStep("do")}},
	}}
	root, _ := editFixture(t, held)
	reasoning := filepath.Join(t.TempDir(), "why.txt")
	if err := os.WriteFile(reasoning, []byte("\nWhy the step exists.\n\nIts second paragraph.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	const check = "go test ./x -run '^TestMiddle$'"
	output, err := planCommand(t, root, "-addstep", "-before", "last", "-title", "Middle step", "-vcmd", check, "-rationale-file", reasoning, "work", "middle")
	if err != nil || !strings.HasPrefix(output, "work/middle: step added; next: ") {
		t.Fatalf("plan -addstep -before: %v\n%s", err, output)
	}
	if output, err := planCommand(t, root, "-addstep", "-title", "Closing step", "-vcmd", check, "-reason", "Why it closes.", "work/closing"); err != nil {
		t.Fatalf("plan -addstep: %v\n%s", err, output)
	}
	after := editedPlan(t, root)
	if got, want := stepIDs(after.Items[0]), "first middle last closing"; got != want {
		t.Fatalf("steps after adding one before the last and one with no -before = %q, want %q", got, want)
	}
	middle := plan.Step{ID: "middle", Title: "Middle step", Status: plan.StatusOpen, Verify: check, Rationale: "Why the step exists.\n\nIts second paragraph."}
	closing := plan.Step{ID: "closing", Title: "Closing step", Status: plan.StatusOpen, Verify: check, Rationale: "Why it closes."}
	if rendered(t, after.Items[0].Steps) != rendered(t, []plan.Step{held.Items[0].Steps[0], middle, held.Items[0].Steps[1], closing}) {
		t.Fatalf("steps after adding two:\n%s", rendered(t, after.Items[0].Steps))
	}
	if rendered(t, after.Items[1]) != rendered(t, held.Items[1]) {
		t.Fatalf("adding a step changed another row:\n%s", rendered(t, after.Items[1]))
	}
	requireNoCheckRan(t, root)

	refusing, written := editFixture(t, held)
	empty := filepath.Join(t.TempDir(), "empty.txt")
	if err := os.WriteFile(empty, []byte(" \n"), 0o600); err != nil {
		t.Fatal(err)
	}
	const usage = "plan -addstep -title <title> -vcmd <cmd> -rationale-file <file> [-before <step-id>] <item-id> <step-id>"
	requireRefused(t, refusing, written, "work/first", "-addstep", "-title", "Again", "-vcmd", check, "-rationale-file", reasoning, "work", "first")
	requireRefused(t, refusing, written, `"absent"`, "-addstep", "-before", "absent", "-title", "New", "-vcmd", check, "-rationale-file", reasoning, "work", "new")
	requireRefused(t, refusing, written, `"absent"`, "-addstep", "-title", "New", "-vcmd", check, "-rationale-file", reasoning, "absent", "new")
	requireRefused(t, refusing, written, "without a verifier", "-addstep", "-title", "New", "-rationale-file", reasoning, "work", "new")
	requireRefused(t, refusing, written, usage, "-addstep", "-vcmd", check, "-rationale-file", reasoning, "work", "new")
	requireRefused(t, refusing, written, usage, "-addstep", "-title", " ", "-vcmd", check, "-rationale-file", reasoning, "work", "new")
	requireRefused(t, refusing, written, usage, "-addstep", "-title", "New", "-vcmd", check, "-rationale-file", empty, "work", "new")
	requireRefused(t, refusing, written, usage, "-addstep", "-title", "New", "-vcmd", check, "work", "new")
	requireRefused(t, refusing, written, "work/ is invalid", "-addstep", "-title", "New", "-vcmd", check, "-rationale-file", reasoning, "work")
	requireOffered(t, refusing, "-addstep", "-addstep -title <t> -vcmd <cmd> -rationale-file <file> [-before <step-id>] <item-id> <step-id>", "with -addstep: before this step id (default: last)",
		"with -add or -addstep: the task title", "with -add or -addstep: the step's verify command")
}

// TestNoteAppendsToARationale holds plan -note to adding the file's paragraph
// to a step's reasoning after a blank line: at the end, or before the closing
// paragraph that opens with Serves: when the reasoning has one, since the
// plan's rationales close with those lines; when two paragraphs open so, the
// later one is the closing. A step with no reasoning takes the paragraph
// alone. Nothing else of the step, its row or another row
// changes, and a step or row the plan lacks, a note that names no step, a
// missing file and an empty one are refused.
func TestNoteAppendsToARationale(t *testing.T) {
	t.Parallel()
	step := func(id, rationale string) plan.Step {
		noted := openStep(id)
		noted.Rationale = rationale
		return noted
	}
	held := plan.Plan{Items: []plan.Item{
		{ID: "work", Title: "Work", Status: plan.StatusOpen, Steps: []plan.Step{
			step("plain", "First paragraph."), step("bare", ""),
			step("closed", "First paragraph.\n\nServes: the loop. Evidence: the test. Batch: host."),
			step("twice", "First paragraph.\n\nServes: an earlier reading.\n\nA later paragraph.\n\nServes: the loop. Evidence: the test."),
		}},
		{ID: "other", Title: "Other", Status: plan.StatusOpen, Steps: []plan.Step{step("plain", "Another row's reasoning.")}},
	}}
	root, _ := editFixture(t, held)
	note := filepath.Join(t.TempDir(), "note.txt")
	if err := os.WriteFile(note, []byte("\nSecond paragraph.\n\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, names := range [][]string{{"work", "plain"}, {"work/bare"}, {"work", "closed"}, {"work/twice"}} {
		output, err := planCommand(t, root, append([]string{"-note", "-rationale-file", note}, names...)...)
		if err != nil || !strings.HasPrefix(output, strings.Join(names, "/")+": reasoning added to; next: ") {
			t.Fatalf("plan -note %v: %v\n%s", names, err, output)
		}
	}
	want := held
	want.Items = []plan.Item{held.Items[0], held.Items[1]}
	want.Items[0].Steps = []plan.Step{
		step("plain", "First paragraph.\n\nSecond paragraph."), step("bare", "Second paragraph."),
		step("closed", "First paragraph.\n\nSecond paragraph.\n\nServes: the loop. Evidence: the test. Batch: host."),
		step("twice", "First paragraph.\n\nServes: an earlier reading.\n\nA later paragraph.\n\nSecond paragraph.\n\nServes: the loop. Evidence: the test."),
	}
	if after := editedPlan(t, root); rendered(t, after) != rendered(t, want) {
		t.Fatalf("plan after four notes:\n%s\nwant\n%s", rendered(t, after), rendered(t, want))
	}
	requireNoCheckRan(t, root)

	refusing, written := editFixture(t, held)
	empty := filepath.Join(t.TempDir(), "empty.txt")
	if err := os.WriteFile(empty, []byte("\n \n"), 0o600); err != nil {
		t.Fatal(err)
	}
	const usage = "plan -note -rationale-file <file> <item-id> <step-id>"
	requireRefused(t, refusing, written, `"absent"`, "-note", "-rationale-file", note, "work", "absent")
	requireRefused(t, refusing, written, `"absent"`, "-note", "-rationale-file", note, "absent", "plain")
	requireRefused(t, refusing, written, usage, "-note", "-rationale-file", empty, "work", "plain")
	requireRefused(t, refusing, written, "-reason gives an edit's words, its reason or reasoning, and -rationale-file a file that holds them", "-note", "work", "plain")
	requireRefused(t, refusing, written, usage, "-note", "-rationale-file", note, "work")
	requireRefused(t, refusing, written, "not found", "-note", "-rationale-file", note, "work/")
	requireRefused(t, refusing, written, "absent.txt", "-note", "-rationale-file", filepath.Join(refusing, "absent.txt"), "work", "plain")
	requireOffered(t, refusing, "-note", "before its closing Serves lines: -note -rationale-file <file> <item-id> <step-id>")
}

// TestShowPrintsARowWithItsRationale holds plan -show to printing one row
// whole and nothing of another: its count with the reason, and each step's
// title, state, what it awaits, check and reasoning. It changes nothing and
// refuses a row the plan lacks. The task -prompt prints for the same step
// leaves the reasoning out, and names the row's scratch folder.
func TestShowPrintsARowWithItsRationale(t *testing.T) {
	t.Parallel()
	held := plan.Plan{Items: []plan.Item{
		{ID: "shown", Title: "Shown row", Status: plan.StatusOpen, Budget: plan.Budget{Nodes: 12, Reason: "It adds a flag."}, Steps: []plan.Step{
			{ID: "first", Title: "First step", Status: plan.StatusOpen, Verify: "go test ./x", Rationale: "Why the first step.\n\nIts second paragraph."},
			{ID: "second", Title: "Second step", Status: "blocked: owner decision", Verify: "go test ./y", DependsOn: []string{"shown/first", "other/do"}, Rationale: "Why the second waits."},
		}},
		{ID: "other", Title: "Other row", Status: plan.StatusOpen, Steps: []plan.Step{
			{ID: "do", Title: "Other step", Status: plan.StatusOpen, Verify: "go test ./z", Rationale: "Another row's reasoning."},
		}},
	}}
	root, written := editFixture(t, held)
	output, err := planCommand(t, root, "-show", "shown")
	const want = `shown | Shown row | nodes 12
count reason: It adds a flag.
-- first | First step | open | awaits []
verify: go test ./x
Why the first step.

Its second paragraph.
-- second | Second step | blocked: owner decision | awaits [shown/first other/do]
verify: go test ./y
Why the second waits.
`
	if err != nil || output != want {
		t.Fatalf("plan -show shown: %v\n%s\nwant\n%s", err, output, want)
	}
	if after := planBytes(t, filepath.Join(root, filepath.FromSlash(plan.Path))); !bytes.Equal(after, written) {
		t.Fatalf("plan -show changed the plan:\n%s", after)
	}
	requireRefused(t, root, written, `"absent"`, "-show", "absent")
	requireRefused(t, root, written, "not found", "-show")
	requireOffered(t, root, "-show", "the reasoning -prompt leaves out: -show <item-id>")

	var prompt bytes.Buffer
	printPromptStep(held.Items[0], held.Items[0].Steps[0], &prompt)
	if text := prompt.String(); !strings.Contains(text, "\nSCRATCH tmp/shown/ ") || strings.Contains(text, "Why the first step.") {
		t.Fatalf("the task prompt names no scratch folder for its row, or carries the reasoning -show prints:\n%s", text)
	}
}

// TestRetireRemovesARowOrAStep holds plan -retire to removing work that will
// not be done, with a stated reason and no check: one step whatever its
// state, where a commit completes only an open one; a whole row with every
// step it has; and a row with its last step. A step that awaited retired work
// keeps its list and waits no longer, as after a completion. The command
// repeats the reason and says no landing records it, and refuses a
// retirement with no reason, and a row or a step the plan lacks.
func TestRetireRemovesARowOrAStep(t *testing.T) {
	t.Parallel()
	blocked := openStep("blocked")
	blocked.Status = "blocked: superseded"
	waiting := openStep("do")
	waiting.DependsOn = []string{"dropped/do", "work/blocked"}
	held := plan.Plan{Items: []plan.Item{
		{ID: "waits", Title: "Waits", Status: plan.StatusOpen, Steps: []plan.Step{waiting}},
		{ID: "work", Title: "Work", Status: plan.StatusOpen, Steps: []plan.Step{blocked, openStep("open")}},
		{ID: "dropped", Title: "Dropped", Status: plan.StatusOpen, Steps: []plan.Step{openStep("do"), openStep("later")}},
		{ID: "single", Title: "Single", Status: plan.StatusOpen, Steps: []plan.Step{openStep("only")}},
	}}
	if _, err := plan.Advance(held, "work", "blocked"); err == nil {
		t.Fatal("a completion removed a step that is not open; the fixture shows no difference from a retirement")
	}
	if item, _, ok := plan.Current(held, worklease.UnassignedRole); !ok || item.ID != "work" {
		t.Fatalf("before any retirement the dispatch is row %q, want the row after the one that waits", item.ID)
	}
	root, _ := editFixture(t, held)
	const reason = "Superseded by the listing."
	for _, reference := range []string{"work/blocked", "dropped", "single/only"} {
		output, err := planCommand(t, root, "-retire", "-reason", reason, reference)
		if err != nil || !strings.HasPrefix(output, reference+": retired; no Landed-Row trailer records it, so the commit's message says why: "+reason+"; next: ") {
			t.Fatalf("plan -retire %s: %v\n%s", reference, err, output)
		}
	}
	after := editedPlan(t, root)
	if got, want := rowIDs(after), "waits work"; got != want {
		t.Fatalf("rows after retiring a step, a row and a row's last step = %q, want %q", got, want)
	}
	if rendered(t, after.Items[0]) != rendered(t, held.Items[0]) || rendered(t, after.Items[1].Steps) != rendered(t, held.Items[1].Steps[1:]) {
		t.Fatalf("rows left after the retirements:\n%s", rendered(t, after.Items))
	}
	if item, step, ok := plan.Current(after, worklease.UnassignedRole); !ok || item.ID != "waits" || step.ID != "do" {
		t.Fatalf("after the retirements the dispatch is %s/%s, want the step that awaited the retired work", item.ID, step.ID)
	}
	requireNoCheckRan(t, root)

	refusing, written := editFixture(t, held)
	const usage = "plan -retire -reason <why> <item-id>[/<step-id>]"
	requireRefused(t, refusing, written, usage, "-retire", "work/blocked")
	requireRefused(t, refusing, written, "not found", "-retire", "-reason", reason)
	requireRefused(t, refusing, written, `"absent"`, "-retire", "-reason", reason, "absent")
	requireRefused(t, refusing, written, `"absent"`, "-retire", "-reason", reason, "work/absent")
	requireRefused(t, refusing, written, `"absent"`, "-retire", "-reason", reason, "single/absent")
	requireRefused(t, refusing, written, "not found", "-retire", "-reason", reason, "work/")
	requireOffered(t, refusing, "-retire", "-retire -reason <why> <item-id>[/<step-id>]",
		"A landing completes an open step by its check and records it in a Landed-Row trailer; a retirement runs no check and leaves no trailer, so the commit's message carries the reason")
}
