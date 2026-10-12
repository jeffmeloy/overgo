// plan edits and dispatches the validated campaign in docs/plan.json.
// Dispatch claims work; the gate verifies, publishes and advances it.
// Status and context inspect state without acquiring work.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/commanddoc"
	"overgo/internal/extent"
	"overgo/internal/plan"
	"overgo/internal/planverify"
	"overgo/internal/webuilane"
	"overgo/internal/worklease"
)

// CLI repository operations resolve from the current worktree directory.
const commandWorktree = "."

func main() {
	flag.CommandLine.Init(os.Args[0], flag.ContinueOnError)
	next := flag.Bool("next", false, "print the top open action")
	prompt := flag.Bool("prompt", false, "print the current step's task: -prompt [-json] [item/step]")
	verify := flag.Bool("verify", false, "run the top open step's verify command; exit code is pass/fail")
	status := flag.Bool("status", false, "one line per item")
	decisions := flag.Bool("decisions", false, "the owner decisions still open, with how long each has waited")
	answer := flag.Bool("answer", false, "record the owner's answer to a decision: -answer <decision-id> -title <one of its options>")
	add := flag.Bool("add", false, "inject a new top-priority task owned by -role: -add <item-id> -title <t> [-before <id>] [-vcmd <cmd>]; or file a document's rows: -add -from <document> [-before <id>]")
	from := flag.String("from", "", "with -add: a plan-shaped JSON document; every row and decision of it is filed, an id the plan holds is refused, and no row the plan holds changes")
	setcount := flag.Bool("setcount", false, "set what a row may still add and why, running no check: -setcount -nodes <n> -reason <why> <item-id>")
	nodes := flag.Int("nodes", 0, "with -setcount: the production syntax nodes the row may still add (0, the default: it may not grow; below zero: it must shrink)")
	reason := flag.String("reason", "", "with -setcount, -addstep, -note or -retire: the edit's words, why the row grows, the step's reasoning or why the work will not be done")
	rationaleFile := flag.String("rationale-file", "", "in place of -reason: a file that holds the edit's words")
	addstep := flag.Bool("addstep", false, "add an open step to a row, last unless -before names a step: -addstep -title <t> -vcmd <cmd> -rationale-file <file> [-before <step-id>] <item-id> <step-id>")
	note := flag.Bool("note", false, "add a paragraph to a step's reasoning, before its closing Serves lines: -note -rationale-file <file> <item-id> <step-id>")
	show := flag.Bool("show", false, "print one row whole, with each step's check, what it awaits and the reasoning -prompt leaves out: -show <item-id>")
	retire := flag.Bool("retire", false, "remove a row, or one step, that will not be worked: -retire -reason <why> <item-id>[/<step-id>]. A landing completes an open step by its check and records it in a Landed-Row trailer; a retirement runs no check and leaves no trailer, so the commit's message carries the reason")
	move := flag.Bool("move", false, "re-rank an open item: -move <item-id> [-before <id>] (default: top of the plan)")
	retitle := flag.Bool("retitle", false, "re-scope an open item and its single step: -retitle <item-id> -title <t>")
	setverify := flag.Bool("setverify", false, "set an existing step's verify: -setverify <item> <step> -vcmd <cmd> (then runs it; exit code is the verdict)")
	jsonFlag := flag.Bool("json", false, "with -next or -prompt: print typed JSON")
	stop := flag.Bool("stop", false, "record a scoped stop until explicit resume: -stop <user-stop|irreversible|external-prereq>: <detail>")
	resumeStop := flag.String("resume-stop", "", "explicitly resume the exact stop ID, preserving its owner and scope")
	maintenanceStop := flag.String("maintenance-stop", "", "authorize one supervised task while the exact stop remains active: <item/step> <reason>")
	stopMode := flag.String("stop-mode", "", "stop scope: all (default), interactive or unattended; resume preserves the recorded scope")
	executionMode := flag.String("mode", "", "execution mode: interactive or unattended; default inherited from the driver")
	title := flag.String("title", "", "with -add or -addstep: the task title")
	class := flag.String("class", "", "with -add: the row class (consolidation; empty is a feature)")
	before := flag.String("before", "", "with -add: insert before this item id (default: top of the plan); with -addstep: before this step id (default: last)")
	verifyCmd := flag.String("vcmd", "", "with -add or -addstep: the step's verify command (a shell command that exits 0 iff accepted)")
	role := flag.String("role", "", "lane role for dispatch (default OVERGO_AUTOMATION_ROLE, then unassigned)")
	worker := flag.String("worker", "", "stable session identity a stop is scoped by (default OVERGO_AUTOMATION_WORKER)")
	handled, err := commanddoc.Plan.Command.ParseCommandLine(flag.CommandLine, os.Stdout)
	if err == nil && !handled {
		err = run(cli{resumeStop: *resumeStop, maintenanceStop: *maintenanceStop, stopMode: *stopMode, mode: *executionMode, json: *jsonFlag, move: *move, retitle: *retitle, next: *next, prompt: *prompt, verify: *verify, status: *status, decisions: *decisions, answer: *answer, add: *add, setverify: *setverify, stop: *stop, title: *title, class: *class, before: *before, verifyCmd: *verifyCmd, role: *role, worker: *worker,
			from: *from, setcount: *setcount, nodes: *nodes, reason: *reason, addstep: *addstep, note: *note, rationaleFile: *rationaleFile, show: *show, retire: *retire}, flag.Args())
	}
	clioptions.MainNamed("plan", func() error { return err })
}

type cli struct {
	resumeStop, maintenanceStop, stopMode, mode  string
	worker                                       string
	next, prompt, verify, status, add, setverify bool
	decisions, answer                            bool
	stop, move, retitle, json                    bool
	title, class, before, verifyCmd, role        string
	setcount, addstep, note, show, retire        bool
	from, reason, rationaleFile                  string
	nodes                                        int
}

func run(c cli, args []string) error {
	if c.stop || c.resumeStop != "" || c.maintenanceStop != "" {
		allowed := cli{stop: c.stop, resumeStop: c.resumeStop, maintenanceStop: c.maintenanceStop, stopMode: c.stopMode, role: c.role, worker: c.worker, json: c.json}
		if c != allowed || c.stop && (c.resumeStop != "" || c.maintenanceStop != "") || c.resumeStop != "" && c.maintenanceStop != "" {
			return errors.New("plan: stop control cannot be combined with another operation")
		}
		return recordStopControlCommand(commandWorktree, c, args, os.Stdout)
	}
	if c.stopMode != "" {
		return errors.New("-stop-mode requires a stop control operation")
	}

	if c.next || c.prompt || c.verify {
		return printDispatch(c, args, os.Stdout)
	}
	if c.mode != "" {
		return errors.New("-mode requires -next, -prompt or -verify")
	}
	document, err := loadCommandPlan()
	if err != nil {
		return err
	}
	role, err := plan.AutomationRole(c.role)
	if err != nil {
		return err
	}
	switch {
	case c.add && c.from != "":
		if len(args) != 0 {
			return errors.New("usage: plan -add -from <document> [-before <id>]")
		}
		filed, err := plan.Load(c.from)
		if err != nil {
			return err
		}
		// Only the document's rows and decisions are taken, and no row the plan
		// holds is edited: what awaits a filed row is the document's to say.
		return mutatePlan(commandWorktree, role, "filed the rows and decisions of "+c.from, func(document plan.Plan) (plan.Plan, error) {
			document.Decisions = append(document.Decisions, filed.Decisions...)
			return insertRows(document, c.before, filed.Items...)
		})
	case c.setcount:
		return editPlan(role, c, args, fmt.Sprintf("count set to %d nodes", c.nodes))
	case c.addstep:
		return editPlan(role, c, args, "step added")
	case c.note:
		return editPlan(role, c, args, "reasoning added to")
	case c.retire:
		return editPlan(role, c, args, "retired; no Landed-Row trailer records it, so the commit's message says why")
	case c.show:
		return printRow(os.Stdout, document, strings.Join(args, "/"))
	case c.add:
		if len(args) != 1 || strings.TrimSpace(c.title) == "" {
			return errors.New("usage: plan -add <item-id> -title <title> [-before <id>] [-vcmd <verify>]")
		}
		return addItem(commandWorktree, args[0], c.title, c.class, c.before, c.verifyCmd, role)
	case c.move:
		if len(args) != 1 {
			return errors.New("usage: plan -move <item-id> [-before <id>]")
		}
		return moveItem(commandWorktree, args[0], c.before, role)
	case c.retitle:
		if len(args) != 1 || strings.TrimSpace(c.title) == "" {
			return errors.New("usage: plan -retitle <item-id> -title <title>")
		}
		return retitleItem(commandWorktree, args[0], c.title, role)
	case c.setverify:
		if len(args) != extent.PairedExtent || strings.TrimSpace(c.verifyCmd) == "" {
			return errors.New("usage: plan -setverify <item-id> <step-id> -vcmd <cmd>")
		}
		if browserVerifyOutsideLane(c.verifyCmd) {
			return errors.New("plan: a browser test is evidence only through cmd/webui-lane (outside it the test skips); name the lane with -run and -leg in the verify")
		}
		return setStepVerify(commandWorktree, args[0], args[1], c.verifyCmd, role)
	case c.status:
		printStatus(document)
		return nil
	case c.decisions:
		printDecisions(os.Stdout, document, time.Now())
		return nil
	case c.answer:
		if len(args) != 1 || strings.TrimSpace(c.title) == "" {
			return errors.New("usage: plan -answer <decision-id> -title <one of its options>")
		}
		return withPlanMutation(commandWorktree, func(document plan.Plan) error {
			answered, err := plan.Answer(document, args[0], c.title)
			if err != nil {
				return err
			}
			if err := savePlanMutation(commandWorktree, answered); err != nil {
				return err
			}
			fmt.Printf("answered %s: %s\n", args[0], c.title)
			return nil
		})
	default:
		return errors.New("one of -next, -prompt, -verify, -status, -show, -decisions, -answer, -add, -addstep, -move, -retitle, -setverify, -setcount, -note, -retire, -stop is required")
	}
}

func withPlanMutation(root string, mutate func(plan.Plan) error) (err error) {
	if mutate == nil {
		return errors.New("plan: mutation callback is required")
	}
	lock, err := authoritylock.Acquire(root)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	document, err := plan.Load(filepath.Join(root, filepath.FromSlash(plan.Path)))
	if err != nil {
		return err
	}
	return mutate(document)
}

// addItem injects a new task as a top-priority item (one step "do"), inserted
// before `before` (or at the top of the plan when empty). This is the mechanical
// "inject a task" operation -- a merge, a fix, or any owner-requested work becomes
// a first-class dispatched/verified/advanced task without hand-editing plan.json.
func addItem(root, id, title, class, before, verifyCmd, role string) error {
	return mutatePlan(root, role, "added item "+id+" (step do)", func(document plan.Plan) (plan.Plan, error) {
		updated, err := insertItem(document, id, title, class, before, verifyCmd)
		if err == nil && role != worklease.UnassignedRole {
			updated.Items[slices.IndexFunc(updated.Items, itemNamed(id))].Owner = role
		}
		return updated, err
	})
}

// moveItem changes item order through the shared mutation owner.
func moveItem(root, id, before, role string) error {
	return mutatePlan(root, role, "moved item "+id, func(document plan.Plan) (plan.Plan, error) {
		return relocateItem(document, id, before)
	})
}

// retitleItem updates the item and its injected single-step title together.
func retitleItem(root, id, title, role string) error {
	return mutatePlan(root, role, "retitled item "+id, func(document plan.Plan) (plan.Plan, error) {
		return rescopeItem(document, id, title)
	})
}

// mutatePlan applies one pure plan mutation, saves it and prints what changed and the next action.
func mutatePlan(root, role, what string, mutation func(plan.Plan) (plan.Plan, error)) error {
	return withPlanMutation(root, func(document plan.Plan) error {
		updated, err := mutation(document)
		if err != nil {
			return err
		}
		if err := savePlanMutation(root, updated); err != nil {
			return err
		}
		action, _ := nextAction(updated, role)
		fmt.Printf("%s; next: %s\n", what, action)
		return nil
	})
}

// rescopeItem is the pure core of retitleItem. No I/O.
func rescopeItem(document plan.Plan, id, title string) (plan.Plan, error) {
	title = strings.TrimSpace(title)
	index := slices.IndexFunc(document.Items, itemNamed(id))
	switch {
	case title == "":
		return plan.Plan{}, errors.New("retitle: the title is empty")
	case index < 0:
		return plan.Plan{}, fmt.Errorf("item %q: no such item", id)
	}
	document.Items[index].Title = title
	if steps := document.Items[index].Steps; len(steps) == 1 && steps[0].ID == "do" {
		steps[0].Title = title
	}
	return document, nil
}

// itemNamed matches a plan item by id.
func itemNamed(id string) func(plan.Item) bool {
	return func(item plan.Item) bool { return item.ID == id }
}

// positionBefore is where an item lands among items: before `before`, or at
// the top when empty.
func positionBefore(items []plan.Item, before string) (top int, err error) {
	if before == "" {
		return top, nil
	}
	if pos := slices.IndexFunc(items, itemNamed(before)); pos >= top {
		return pos, nil
	}
	return top, fmt.Errorf("-before %q: no such item", before)
}

// relocateItem is the pure core of moveItem: returns a plan with the item
// removed from its position and reinserted before `before` (or at the top
// when empty). Moving an item before itself is the identity. No I/O.
func relocateItem(document plan.Plan, id, before string) (plan.Plan, error) {
	from := slices.IndexFunc(document.Items, itemNamed(id))
	switch {
	case id == before:
		return document, nil
	case from < 0:
		return plan.Plan{}, fmt.Errorf("item %q: no such item", id)
	}
	rest := slices.DeleteFunc(slices.Clone(document.Items), itemNamed(id))
	pos, err := positionBefore(rest, before)
	if err != nil {
		return plan.Plan{}, err
	}
	document.Items = slices.Insert(rest, pos, document.Items[from])
	return document, nil
}

// insertItem is the pure core of addItem: returns a plan with a new open item
// (one step "do") inserted before `before` (or at the top when empty), and,
// when awaited, the campaign closeout waiting for it, as the structure policy
// requires. A merge's own row is not awaited: the merge completes it, so the
// merged plan equals the first-parent plan. No I/O.
func insertItem(document plan.Plan, id, title, class, before, verifyCmd string) (plan.Plan, error) {
	document, err := insertRows(document, before, plan.Item{
		ID: id, Title: title, Class: class, Status: "open",
		Steps: []plan.Step{{ID: "do", Title: title, Status: "open", Verify: verifyCmd}},
	})
	if closeout := slices.IndexFunc(document.Items, itemNamed("campaign-closeout")); class != plan.ClassMerge && closeout >= 0 {
		steps := slices.Clone(document.Items[closeout].Steps)
		for index := range steps {
			steps[index].DependsOn = append(slices.Clone(steps[index].DependsOn), id+"/do")
		}
		document.Items[closeout].Steps = steps
	}
	return document, err
}

// insertRows is the one way this command puts rows into a plan: together and
// in their order, before `before` or at the top when empty, never under an
// id the plan holds. It edits no row the plan had. No I/O.
func insertRows(document plan.Plan, before string, rows ...plan.Item) (plan.Plan, error) {
	pos, err := positionBefore(document.Items, before)
	for _, row := range rows {
		if slices.ContainsFunc(document.Items, itemNamed(row.ID)) {
			return plan.Plan{}, fmt.Errorf("item %q already exists", row.ID)
		}
	}
	if err != nil {
		return plan.Plan{}, err
	}
	document.Items = slices.Insert(slices.Clone(document.Items), pos, rows...)
	return document, nil
}

// stepAt finds the row an edit names and the place of the step it names
// there; for an edit that names no step, the place after the row's last.
func stepAt(document plan.Plan, itemID, stepID string, named bool) (item, step int, err error) {
	if item = slices.IndexFunc(document.Items, itemNamed(itemID)); item < 0 {
		return item, step, fmt.Errorf("item %q not found", itemID)
	}
	steps := document.Items[item].Steps
	if step = slices.IndexFunc(steps, func(candidate plan.Step) bool { return candidate.ID == stepID }); !named {
		step = len(steps)
	} else if step < 0 {
		err = fmt.Errorf("step %q not found in %q", stepID, itemID)
	}
	return item, step, err
}

// assignVerify is the pure core of setStepVerify: returns a plan with the named
// step's Verify replaced, or an error if the item/step is absent.
func assignVerify(document plan.Plan, itemID, stepID, cmd string) (plan.Plan, error) {
	item, step, err := stepAt(document, itemID, stepID, true)
	if err != nil {
		return plan.Plan{}, err
	}
	document.Items[item].Steps[step].Verify = strings.TrimSpace(cmd)
	return document, nil
}

// rationaleClosing opens the paragraph a rationale closes with, its Serves,
// Evidence and Batch lines: a note goes before it, so they stay last.
const rationaleClosing = "\n\nServes:"

// editUsage is what the edits of a row or a step take.
const editUsage = `usage, with every flag before the names, where Go's flag parsing stops:
  plan -setcount -nodes <n> -reason <why> <item-id>
  plan -addstep -title <title> -vcmd <cmd> -rationale-file <file> [-before <step-id>] <item-id> <step-id>
  plan -note -rationale-file <file> <item-id> <step-id>
  plan -retire -reason <why> <item-id>[/<step-id>]
-reason gives an edit's words, its reason or reasoning, and -rationale-file a file that holds them`

// editPlan makes one of the edits of a row or a step that one-use programs
// used to make, through the plan's lock and its one writer, running no check.
// The names after the flags are the row and, for a step, the step, as two
// words or joined by a slash.
//
// A retirement is not a completion. A landing completes one open step whose
// check passed, and its commit says so in a Landed-Row trailer (plan.Advance).
// A retirement removes a step whatever its state, or a row with every step it
// has; it runs no check and leaves no trailer, so only the message of the
// commit that carries the plan says why. A step that awaited the work waits
// no longer in either case.
func editPlan(role string, c cli, args []string, did string) error {
	reference, words := strings.Join(args, "/"), strings.TrimSpace(c.reason)
	if c.rationaleFile != "" {
		text, err := os.ReadFile(c.rationaleFile)
		if err != nil {
			return err
		}
		words = strings.TrimSpace(string(text))
	}
	itemID, stepID, named := strings.Cut(reference, "/")
	if words == "" || c.note && !named || c.setcount && named || c.addstep && strings.TrimSpace(c.title) == "" {
		return errors.New(editUsage)
	}
	if c.retire {
		did += ": " + words
	}
	return mutatePlan(commandWorktree, role, reference+": "+did, func(document plan.Plan) (plan.Plan, error) {
		// A new step is placed by -before; every other edit by the step it names.
		place, placed := stepID, named
		if c.addstep {
			place, placed = c.before, c.before != ""
		}
		item, step, err := stepAt(document, itemID, place, placed)
		if err != nil {
			return document, err
		}
		row := &document.Items[item]
		switch {
		case c.setcount:
			row.Budget.Nodes, row.Budget.Reason = c.nodes, words
		case c.addstep:
			row.Steps = slices.Insert(row.Steps, step, plan.Step{ID: stepID, Title: c.title, Status: plan.StatusOpen, Verify: c.verifyCmd, Rationale: words})
		case c.note:
			rationale := &row.Steps[step].Rationale
			cut := strings.LastIndex(*rationale, rationaleClosing)
			if cut < 0 {
				cut = len(*rationale)
			}
			*rationale = strings.TrimSpace((*rationale)[:cut] + "\n\n" + words + (*rationale)[cut:])
		case c.retire && named && len(row.Steps) > 1:
			row.Steps = slices.Delete(row.Steps, step, step+1)
		case c.retire:
			// The row goes whole, or with the last step it had.
			document.Items = slices.Delete(document.Items, item, item+1)
		}
		return document, nil
	})
}

// printRow prints one row whole: its count with the reason, and each step's
// state, what it awaits, its check and its reasoning.
func printRow(output io.Writer, document plan.Plan, id string) error {
	at, _, err := stepAt(document, id, "", false)
	if err != nil {
		return err
	}
	item := document.Items[at]
	fmt.Fprintf(output, "%s | %s | nodes %d\ncount reason: %s\n", item.ID, item.Title, item.Budget.Nodes, item.Budget.Reason)
	for _, step := range item.Steps {
		fmt.Fprintf(output, "-- %s | %s | %s | awaits %v\nverify: %s\n%s\n", step.ID, step.Title, step.Status, step.DependsOn, step.Verify, step.Rationale)
	}
	return nil
}

// setStepVerify records an existing step's verify command (closing the gap that
// forced hand-editing docs/plan.json), then runs it so an unrunnable command --
// an unquoted shell metachar, a bad -run pattern -- is caught at set-time rather
// than at the next advance.
func setStepVerify(root, itemID, stepID, cmd, role string) error {
	return withPlanMutation(root, func(document plan.Plan) error {
		updated, err := assignVerify(document, itemID, stepID, cmd)
		if err != nil {
			return err
		}
		it, st, current := plan.Current(updated, role)
		if err := savePlanMutation(root, updated); err != nil {
			return err
		}
		fmt.Printf("set verify for %s/%s: %s\n", itemID, stepID, strings.TrimSpace(cmd))
		if current && it.ID == itemID && st.ID == stepID {
			return runVerify(it, st)
		}
		return nil
	})
}

// printDecisions lists the owner decisions still open, oldest first as the
// plan holds them, with the days each has waited: a decision that sits
// stalls the rows it blocks.
func printDecisions(output io.Writer, document plan.Plan, now time.Time) {
	for _, decision := range document.Decisions {
		if decision.Answer != "" {
			continue
		}
		asked, _ := time.Parse(time.DateOnly, decision.Asked)
		days := int(now.Sub(asked).Hours() / hoursPerDay)
		fmt.Fprintf(output, "%s (asked %s, %d days): %s\n", decision.ID, decision.Asked, days, decision.Question)
		for _, option := range decision.Options {
			mark := " "
			if option == decision.Recommendation {
				mark = "*"
			}
			fmt.Fprintf(output, "  %s %s\n", mark, option)
		}
		if len(decision.Blocks) != 0 {
			fmt.Fprintf(output, "  blocks: %s\n", strings.Join(decision.Blocks, ", "))
		}
	}
}

// hoursPerDay turns a wait into days.
const hoursPerDay = 24

func printStatus(document plan.Plan) {
	for _, entry := range document.Items {
		open := 0
		for _, step := range entry.Steps {
			if step.Status == plan.StatusOpen {
				open++
			}
		}
		fmt.Printf("%-30s %-24s %d open rows  %s\n", entry.ID, entry.Status, open, entry.Title)
	}
}

// nextAction: the one-line form of the current step, rendered from the
// dispatch's fields.
func nextAction(document plan.Plan, role string) (string, bool) {
	dispatch := plan.DispatchOf(document, role)
	if dispatch.Complete {
		return "", false
	}
	return dispatch.Line, true
}

// printPrompt emits the self-contained, non-negotiable task for the current
// step. The loop feeds THIS to the agent; the agent does not author it.
func printPrompt(document plan.Plan, role string, output io.Writer) {
	it, st, ok := plan.Current(document, role)
	if !ok {
		fmt.Fprintln(output, "PLAN COMPLETE: every item is done. Stop and tell the user.")
		return
	}
	printPromptStep(it, st, output)
}

// locateStep finds the dispatched row in the document; a complete dispatch
// or a row the document no longer holds is not found.
func locateStep(document plan.Plan, dispatch plan.Dispatch) (plan.Item, plan.Step, bool) {
	if dispatch.Complete {
		return plan.Item{}, plan.Step{}, false
	}
	for _, it := range document.Items {
		if it.ID != dispatch.Item {
			continue
		}
		if dispatch.Step == "." {
			return it, plan.Step{ID: "."}, true
		}
		for _, st := range it.Steps {
			if st.ID == dispatch.Step {
				return it, st, true
			}
		}
	}
	return plan.Item{}, plan.Step{}, false
}

// printPromptStep renders the task for one located row.
func printPromptStep(it plan.Item, st plan.Step, output io.Writer) {
	verify := st.Verify
	if strings.TrimSpace(verify) == "" {
		verify = "MISSING -- add a runnable step.verify before implementation"
	}
	fmt.Fprintf(output, "TASK %s/%s\n%s\nVERIFY %s\n", it.ID, st.ID, st.Title, verify)
	if len(st.DependsOn) != 0 {
		fmt.Fprintf(output, "PREREQUISITES %s\n", strings.Join(st.DependsOn, commandWordSeparator))
	}
	fmt.Fprintf(output, `COMMIT go run ./cmd/loop -land %[1]s/%[2]s -message-file <msg> (the small gate over the dirty tree, the landing confirmed from git)
SCRATCH tmp/%[1]s/ for this row's probes and documents only
FINDINGS park off-scope work: go run ./cmd/finding -h
RULES skill.md; only this task; port-first; gate advances atomically; then rerun plan -prompt.
STOP only a user stop pauses the loop (plan -stop <reason>); rationale stays in docs/plan.json and landed rows in git (Landed-Row trailers).
`, it.ID, st.ID)
	if st.VerificationBatch != nil {
		for _, checkpoint := range st.VerificationBatch.Checkpoints {
			fmt.Fprintf(output, "BATCH ACCEPTANCE %s: %s\n", checkpoint.ID, checkpoint.Verify)
		}
		fmt.Fprintln(output, "BATCH declarations add required acceptance; full gating and parent completion remain mandatory.")
	}
}

// browserVerifyOutsideLane: a verify naming a browser acceptance test without the lane runner that makes it run.
func browserVerifyOutsideLane(command string) bool {
	return (strings.Contains(command, webuilane.BrowserTestPrefix) || strings.Contains(command, webuilane.ModelJourneyPrefix)) && !strings.Contains(command, "cmd/webui-lane")
}

// runVerify executes the step's verify command; its exit code is the verdict.
func runVerify(it plan.Item, st plan.Step) error {
	if strings.TrimSpace(st.Verify) == "" {
		return fmt.Errorf("no verify defined for %s/%s -- add a runnable step.verify (exits 0 iff accepted) before advancing", it.ID, st.ID)
	}
	if st.VerificationBatch != nil {
		for _, checkpoint := range st.VerificationBatch.Checkpoints {
			fmt.Fprintf(os.Stderr, "plan verify %s/%s checkpoint %s: %s\n", it.ID, st.ID, checkpoint.ID, checkpoint.Verify)
			if _, err := planverify.Execute(context.Background(), commandWorktree, checkpoint.Verify, nil); err != nil {
				return fmt.Errorf("verify %s/%s checkpoint %s: %w", it.ID, st.ID, checkpoint.ID, err)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "plan verify %s/%s: %s\n", it.ID, st.ID, st.Verify)
	class, err := planverify.Execute(context.Background(), commandWorktree, st.Verify, nil)
	if err != nil {
		return fmt.Errorf("verify %s/%s: %w -- run the named oracle against its real prerequisite or record a recorded stop", it.ID, st.ID, err)
	}
	fmt.Fprintf(os.Stderr, "plan verify %s/%s: PASS verdict=%s\n", it.ID, st.ID, class)
	return nil
}

// All command paths use the same document and campaign validation sequence.
func loadCommandPlan() (plan.Plan, error) {
	document, err := plan.Load("")
	if err != nil {
		return document, err
	}
	if err := plan.Validate(document); err != nil {
		return document, err
	}
	return document, nil
}
