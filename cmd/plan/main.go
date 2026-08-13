// plan: campaign dispatch bookkeeping (owner directive 2026-08-09: the turn
// is the plan). Reads docs/plan.json via internal/plan -- the machine-readable
// open-work surface -- and both dispatches and ENFORCES the loop protocol:
//
//	plan -next                     print the top open action (one line)
//	plan -prompt                   print the generated, self-contained task for
//	                               the top open step (this is what the loop
//	                               feeds the agent -- NOT free text it rewrites)
//	plan -verify                   run the top open step's acceptance command
//	                               (step.verify); exit code is pass/fail
//	plan -advance <item> <step>    remove a completed step -- REFUSED unless its
//	                               verify command exits 0 (step "." closes the
//	                               item). -force <reason> overrides loudly.
//	plan -status                   one line per item
//
// Enforcement rationale (owner 2026-08-11, after a session drifted off-plan for
// ~13 commits with zero -advance): "done" must be machine-checked, not
// self-declared prose, and the loop task must be GENERATED from the plan so the
// agent cannot substitute its own agenda. Every step carries a runnable
// `verify` command; -advance gates on it; commit-gate binds each commit to the
// active step (internal/plan.Current is the shared source of truth).
package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"overgo/internal/plan"
	"overgo/internal/testevidence"
)

func main() {
	next := flag.Bool("next", false, "print the top open action")
	prompt := flag.Bool("prompt", false, "print the generated self-contained task for the top open step")
	verify := flag.Bool("verify", false, "run the top open step's verify command; exit code is pass/fail")
	status := flag.Bool("status", false, "one line per item")
	advance := flag.Bool("advance", false, "verify and remove <item> <step> from the open-work queue")
	add := flag.Bool("add", false, "inject a new top-priority task: -add <item-id> -title <t> [-before <id>] [-verify <cmd>]")
	setverify := flag.Bool("setverify", false, "set an existing step's verify: -setverify <item> <step> -vcmd <cmd> (then runs it; exit code is the verdict)")
	stop := flag.Bool("stop", false, "record a legitimate loop stop: -stop <user-stop|irreversible|external-prereq>: <detail>")
	force := flag.String("force", "", "with -advance: skip verify, REQUIRES a reason (logged loudly)")
	title := flag.String("title", "", "with -add: the task title")
	before := flag.String("before", "", "with -add: insert before this item id (default: top of the plan)")
	verifyCmd := flag.String("vcmd", "", "with -add: the step's verify command (a shell command that exits 0 iff accepted)")
	flag.Parse()
	if err := run(cli{next: *next, prompt: *prompt, verify: *verify, status: *status, advance: *advance, add: *add, setverify: *setverify, stop: *stop, force: *force, title: *title, before: *before, verifyCmd: *verifyCmd}, flag.Args()); err != nil {
		fmt.Fprintf(os.Stderr, "plan: %v\n", err)
		os.Exit(1)
	}
}

type cli struct {
	next, prompt, verify, status, advance, add, setverify, stop bool
	force, title, before, verifyCmd                             string
}

func run(c cli, args []string) error {
	document, err := plan.Load("")
	if err != nil {
		return err
	}
	switch {
	case c.add:
		if len(args) != 1 || strings.TrimSpace(c.title) == "" {
			return errors.New("usage: plan -add <item-id> -title <title> [-before <id>] [-vcmd <verify>]")
		}
		return addItem(document, args[0], c.title, c.before, c.verifyCmd)
	case c.setverify:
		if len(args) != 2 || strings.TrimSpace(c.verifyCmd) == "" {
			return errors.New("usage: plan -setverify <item-id> <step-id> -vcmd <cmd>")
		}
		return setStepVerify(document, args[0], args[1], c.verifyCmd)
	case c.stop:
		return recordStop(strings.Join(args, " "))
	case c.advance:
		if len(args) != 2 {
			return errors.New("usage: plan -advance <item-id> <step-id|.>")
		}
		return advanceStep(document, args[0], args[1], c.force)
	case c.status:
		printStatus(document)
		return nil
	case c.prompt:
		printPrompt(document)
		return nil
	case c.verify:
		it, st, ok := plan.Current(document)
		if !ok {
			fmt.Println("plan complete: nothing to verify")
			return nil
		}
		return runVerify(it, st)
	case c.next:
		action, open := nextAction(document)
		if !open {
			fmt.Println("plan complete: every item is done")
			return nil
		}
		fmt.Println(action)
		return nil
	default:
		return errors.New("one of -next, -prompt, -verify, -status, -advance is required")
	}
}

// addItem injects a new task as a top-priority item (one step "do"), inserted
// before `before` (or at the top of the plan when empty). This is the mechanical
// "inject a task" operation -- a merge, a fix, or any owner-requested work becomes
// a first-class dispatched/verified/advanced task without hand-editing plan.json.
func addItem(document plan.Plan, id, title, before, verifyCmd string) error {
	updated, err := insertItem(document, id, title, before, verifyCmd)
	if err != nil {
		return err
	}
	if err := plan.Save("", updated); err != nil {
		return err
	}
	action, _ := nextAction(updated)
	fmt.Printf("added item %s (step do); next: %s\n", id, action)
	return nil
}

// insertItem is the pure core of addItem: returns a plan with a new open item
// (one step "do") inserted before `before` (or at the top when empty). No I/O.
func insertItem(document plan.Plan, id, title, before, verifyCmd string) (plan.Plan, error) {
	for _, it := range document.Items {
		if it.ID == id {
			return plan.Plan{}, fmt.Errorf("item %q already exists", id)
		}
	}
	item := plan.Item{
		ID: id, Title: title, Status: "open",
		Steps: []plan.Step{{ID: "do", Title: title, Status: "open", Verify: verifyCmd}},
	}
	pos := 0
	if before != "" {
		pos = -1
		for i, it := range document.Items {
			if it.ID == before {
				pos = i
				break
			}
		}
		if pos < 0 {
			return plan.Plan{}, fmt.Errorf("-before %q: no such item", before)
		}
	}
	out := make([]plan.Item, 0, len(document.Items)+1)
	out = append(out, document.Items[:pos]...)
	out = append(out, item)
	out = append(out, document.Items[pos:]...)
	document.Items = out
	return document, nil
}

// assignVerify is the pure core of setStepVerify: returns a plan with the named
// step's Verify replaced, or an error if the item/step is absent.
func assignVerify(document plan.Plan, itemID, stepID, cmd string) (plan.Plan, error) {
	for i := range document.Items {
		if document.Items[i].ID != itemID {
			continue
		}
		for j := range document.Items[i].Steps {
			if document.Items[i].Steps[j].ID == stepID {
				document.Items[i].Steps[j].Verify = strings.TrimSpace(cmd)
				return document, nil
			}
		}
		return plan.Plan{}, fmt.Errorf("step %q not found in %q", stepID, itemID)
	}
	return plan.Plan{}, fmt.Errorf("item %q not found", itemID)
}

// setStepVerify records an existing step's verify command (closing the gap that
// forced hand-editing docs/plan.json), then runs it so an unrunnable command --
// an unquoted shell metachar, a bad -run pattern -- is caught at set-time rather
// than at the next advance.
func setStepVerify(document plan.Plan, itemID, stepID, cmd string) error {
	updated, err := assignVerify(document, itemID, stepID, cmd)
	if err != nil {
		return err
	}
	if err := plan.Save("", updated); err != nil {
		return err
	}
	fmt.Printf("set verify for %s/%s: %s\n", itemID, stepID, strings.TrimSpace(cmd))
	it, st, ok := plan.Current(updated)
	if ok && it.ID == itemID && st.ID == stepID {
		return runVerify(it, st)
	}
	return nil
}

var validStopReasons = []string{"user-stop", "irreversible", "external-prereq"}

// validateStop returns nil iff reason begins with a legitimate stop tag -- the
// only three reasons that justify ending a turn without plan progress. A
// self-invented "checkpoint" or "should I continue?" is not among them.
func validateStop(reason string) error {
	reason = strings.TrimSpace(reason)
	for _, v := range validStopReasons {
		if reason == v || strings.HasPrefix(reason, v+":") {
			return nil
		}
	}
	return fmt.Errorf("invalid stop reason %q -- must begin with one of: user-stop: / irreversible: / external-prereq: <detail>", reason)
}

// recordStop writes a valid stop marker at the current HEAD; the stop-gate reads
// it to allow a legitimate turn end. Progress on any later turn supersedes it.
func recordStop(reason string) error {
	if err := validateStop(reason); err != nil {
		return err
	}
	head := "unknown"
	if out, err := exec.Command("git", "rev-parse", "HEAD").Output(); err == nil {
		head = strings.TrimSpace(string(out))
	}
	payload := fmt.Sprintf("{\"reason\":%q,\"head\":%q}\n", strings.TrimSpace(reason), head)
	if err := os.WriteFile("docs/plan_stop.json", []byte(payload), 0o644); err != nil {
		return err
	}
	fmt.Printf("recorded stop: %s\n", strings.TrimSpace(reason))
	return nil
}

func printStatus(document plan.Plan) {
	for _, entry := range document.Items {
		fmt.Printf("%-22s %-6s %d open %s\n", entry.ID, entry.Status, len(entry.Steps), entry.Title)
	}
}

// nextAction: the one-line form of the current step.
func nextAction(document plan.Plan) (string, bool) {
	it, st, ok := plan.Current(document)
	if !ok {
		return "", false
	}
	if st.ID == "." {
		return fmt.Sprintf("%s: %s -- open the rung (define its steps)", it.ID, it.Title), true
	}
	return fmt.Sprintf("%s / %s: %s -- %s", it.ID, st.ID, it.Title, st.Title), true
}

// printPrompt emits the self-contained, non-negotiable task for the current
// step. The loop feeds THIS to the agent; the agent does not author it.
func printPrompt(document plan.Plan) {
	it, st, ok := plan.Current(document)
	if !ok {
		fmt.Println("PLAN COMPLETE: every item is done. Stop and tell the user.")
		return
	}
	doctrine := document.Doctrine
	if len(doctrine) > 700 {
		doctrine = doctrine[:700] + " ...[see docs/plan.json for the full doctrine]"
	}
	verify := st.Verify
	if strings.TrimSpace(verify) == "" {
		verify = "(NONE DEFINED -- you MUST add a runnable step.verify that exits 0 iff this step's\n" +
			"         acceptance holds, to docs/plan.json, before this step can be advanced.)"
	}
	fmt.Printf(`=== PLAN TASK (generated -- do exactly this step, nothing else) ===
Campaign: %s
Item:  %s -- %s
Step:  %s -- %s

BINDING DOCTRINE (port-first):
%s

PROTOCOL -- no deviation:
  1. Do ONLY this step. Port from adaptive_new first, verify against its goldens.
  2. Do NOT start another step, act on a finding, or refactor off to the side. A
     finding goes to docs/findings.json; it becomes work ONLY by later appearing
     here as the top step -- never by you acting on it now.
  3. If this step is wrong, blocked, or you disagree with it: STOP and tell the
     user. Do NOT substitute your own work for the dispatched step.
  4. Commit ONLY via the plan-bound gate:
       go run ./cmd/gate -plan %s/%s -message-file <msg> -paths <csv>
     The gate REFUSES any commit whose -plan is not this active step.
  5. "Done" means: 'go run ./cmd/plan -verify' exits 0 (it runs this step's
     acceptance command below), THEN 'go run ./cmd/plan -advance %s %s'.
  6. After advancing, re-rank/refactor the plan from the result (or user input),
     then 'go run ./cmd/plan -prompt' for the next task. Repeat until complete.

VERIFY (this step's machine-checked acceptance):
  %s
=== END TASK ===
`, document.Campaign, it.ID, it.Title, st.ID, st.Title, doctrine, it.ID, st.ID, it.ID, st.ID, verify)
}

// runVerify executes the step's verify command; its exit code is the verdict.
func runVerify(it plan.Item, st plan.Step) error {
	if strings.TrimSpace(st.Verify) == "" {
		return fmt.Errorf("no verify defined for %s/%s -- add a runnable step.verify (exits 0 iff accepted) before advancing", it.ID, st.ID)
	}
	fmt.Fprintf(os.Stderr, "plan verify %s/%s: %s\n", it.ID, st.ID, st.Verify)
	var buf bytes.Buffer
	shell, err := verificationShell()
	if err != nil {
		return fmt.Errorf("verify %s/%s: %w", it.ID, st.ID, err)
	}
	cmd := exec.Command(shell, "-c", st.Verify)
	cmd.Stdout = io.MultiWriter(os.Stdout, &buf)
	cmd.Stderr = io.MultiWriter(os.Stderr, &buf)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("verify FAILED for %s/%s: %w", it.ID, st.ID, err)
	}
	if err := testevidence.VerifyOutput(st.Verify, buf.String()); err != nil {
		return fmt.Errorf("verify VACUOUS for %s/%s: %v -- a skip is NOT a pass; run the oracle against the real prerequisite (on hardware / with the fixture) or record an honest stop, but do not advance on unverified parity", it.ID, st.ID, err)
	}
	fmt.Fprintf(os.Stderr, "plan verify %s/%s: PASS\n", it.ID, st.ID)
	return nil
}

func verificationShell() (string, error) {
	if shell, err := exec.LookPath("sh"); err == nil {
		return shell, nil
	}
	if runtime.GOOS == "windows" {
		for _, path := range []string{
			`C:\Program Files\Git\bin\bash.exe`,
			`C:\Program Files\Git\usr\bin\bash.exe`,
		} {
			if info, err := os.Stat(filepath.Clean(path)); err == nil && !info.IsDir() {
				return path, nil
			}
		}
	}
	return "", fmt.Errorf("POSIX shell unavailable")
}

// vacuousVerify reports why a 0-exit verify is NOT real evidence, or "" when it
// is. A skipped test still exits 0, so without this a capability whose golden /
// fixture / hardware is absent advances UNVERIFIED -- the exact hole that let
// the un0 batch regression through tightening's gate (the golden was
// UNAVAILABLE on that worktree, the test skipped, exit stayed 0). It keys on the
// repo's existing conventions: an absent prerequisite emits UNAVAILABLE and/or
// "parity NOT verified", and an empty/over-filtered run emits "[no test(s)...]".
// Environment-gated skips that DID run real assertions elsewhere still print a
// package "ok" without these markers and are unaffected.

func advanceStep(document plan.Plan, itemID, stepID, force string) error {
	for i := range document.Items {
		if document.Items[i].ID != itemID {
			continue
		}
		if stepID != "." {
			targetIndex := -1
			for j := range document.Items[i].Steps {
				if document.Items[i].Steps[j].ID == stepID {
					targetIndex = j
					break
				}
			}
			if targetIndex < 0 {
				return fmt.Errorf("step %q not found in %q", stepID, itemID)
			}
			if err := gateAdvance(document.Items[i], document.Items[i].Steps[targetIndex], force); err != nil {
				return err
			}
			document = removeCompleted(document, i, targetIndex)
			return finishAdvance(document, itemID, stepID)
		}
		if err := gateAdvance(document.Items[i], plan.Step{ID: "."}, force); err != nil {
			return err
		}
		document = removeCompleted(document, i, -1)
		return finishAdvance(document, itemID, stepID)
	}
	return fmt.Errorf("item %q not found", itemID)
}

// removeCompleted keeps the persisted plan an open-work queue. A negative
// step index removes the whole item; removing its last step does the same.
func removeCompleted(document plan.Plan, itemIndex, stepIndex int) plan.Plan {
	if stepIndex >= 0 {
		steps := document.Items[itemIndex].Steps
		document.Items[itemIndex].Steps = append(steps[:stepIndex], steps[stepIndex+1:]...)
		if len(document.Items[itemIndex].Steps) > 0 {
			return document
		}
	}
	document.Items = append(document.Items[:itemIndex], document.Items[itemIndex+1:]...)
	return document
}

// gateAdvance refuses the advance unless the step's verify passes, or a -force
// reason is given (logged loudly so an override is never silent).
func gateAdvance(it plan.Item, st plan.Step, force string) error {
	if force != "" {
		fmt.Fprintf(os.Stderr, "plan: WARNING -force advance of %s/%s (verify SKIPPED) -- reason: %s\n", it.ID, st.ID, force)
		return nil
	}
	return runVerify(it, st)
}

func finishAdvance(document plan.Plan, itemID, stepID string) error {
	if err := plan.Save("", document); err != nil {
		return err
	}
	action, open := nextAction(document)
	if !open {
		fmt.Printf("advanced %s/%s; PLAN COMPLETE\n", itemID, stepID)
		return nil
	}
	fmt.Printf("advanced %s/%s; next: %s\n", itemID, stepID, action)
	return nil
}
