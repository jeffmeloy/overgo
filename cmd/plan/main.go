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
//	plan -context                  emit one typed JSON grounding payload for
//	                               the current task and worktree
//	plan -record-lease <file>      record an owner-approved advisory worktree lease
//	plan -lease-report             report active lease conflicts and reservations
//	plan -advance <item> <step>    refused compatibility flag: only cmd/gate may
//	                               atomically verify, commit, and prune a row
//	plan -status                   one line per item
//	plan -prepare-merge <ref>      snapshot and prepare a gated semantic merge
//
// Enforcement rationale (owner 2026-08-11, after a session drifted off-plan for
// ~13 commits with zero -advance): "done" must be machine-checked, not
// self-declared prose, and the loop task must be GENERATED from the plan so the
// agent cannot substitute its own agenda. Every step carries a runnable
// `verify` command; commit-gate binds each commit to the active step and prunes
// it in that commit (internal/plan.Current is the shared source of truth).
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/planverify"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
)

func main() {
	next := flag.Bool("next", false, "print the top open action")
	frontier := flag.Bool("frontier", false, "print every dispatchable row (the ready frontier) and validate live worktree leases against it")
	judgeEfficiency := flag.String("judge-efficiency", "", "judge an interaction-efficiency claim JSON ({candidate, baseline, tradeoff?}); exit code is the verdict")
	admitProposalFlag := flag.String("admit-proposal", "", "admit one typed steering proposal from a JSON spec into the store and the plan")
	history := flag.String("history", "", "print attempt history from the store: aggregates and recent attempts (pass a plan item id, or all)")
	prompt := flag.Bool("prompt", false, "print the generated self-contained task for the top open step")
	verify := flag.Bool("verify", false, "run the top open step's verify command; exit code is pass/fail")
	status := flag.Bool("status", false, "one line per item")
	contextJSON := flag.Bool("context", false, "emit one typed JSON grounding payload for the current automation task")
	recordLease := flag.String("record-lease", "", "record an advisory worktree lease from a JSON file")
	grantExploration := flag.String("grant-exploration", "", "commit an externally-issued exploration budget token from a JSON file")
	chargeExploration := flag.String("charge-exploration", "", "commit one experiment spend against an exploration grant from a JSON file")
	recordLeaseOutcome := flag.String("record-lease-outcome", "", "record measured outcome JSON for an exercised worktree lease")
	recordExperiment := flag.String("record-experiment", "", "commit one experiment lifecycle transition from a JSON spec (state, experiment, evidence, prior)")
	leaseReport := flag.Bool("lease-report", false, "emit active worktree leases and resource/conflict advice as JSON")
	retireLegacyLeases := flag.Int("retire-legacy-leases", -1, "retire pre-contract work-lease aliases through reviewed compare-and-set; the value is the reviewed expected count")
	localitySchedule := flag.String("schedule-locality", "", "schedule a JSON worker/artifact request against OvergoDB locations")
	cpuCapacity := flag.Int("cpu-capacity", 0, "with -lease-report: available CPU threads (0 unknown)")
	ramCapacity := flag.Int("ram-capacity-gib", 0, "with -lease-report: available host RAM GiB (0 unknown)")
	vramCapacity := flag.Int("vram-capacity-gib", 0, "with -lease-report: available VRAM GiB (0 unknown)")
	advance := flag.Bool("advance", false, "retired: cmd/gate atomically commits and prunes the current row")
	bindCensus := flag.Bool("bind-census", false, "bind the campaign baseline to closure/census/latest in OvergoDB")
	pruneDone := flag.Bool("prune-done", false, "retired: direct plan pruning is refused")
	add := flag.Bool("add", false, "inject a new top-priority task owned by -role: -add <item-id> -title <t> [-before <id>] [-verify <cmd>]")
	setverify := flag.Bool("setverify", false, "set an existing step's verify: -setverify <item> <step> -vcmd <cmd> (then runs it; exit code is the verdict)")
	prepareMergeFlag := flag.String("prepare-merge", "", "snapshot a ref and prepare a gated merge with semantic plan and compatibility regeneration")
	stop := flag.Bool("stop", false, "record a legitimate loop stop: -stop <user-stop|irreversible|external-prereq>: <detail>")
	contain := flag.String("contain", "", "record typed lane containment: -contain <reason-code> -lane <lane> <detail>")
	lane := flag.String("lane", "", "lane affected by -contain")
	_ = flag.String("force", "", "retired with -advance")
	title := flag.String("title", "", "with -add: the task title")
	before := flag.String("before", "", "with -add: insert before this item id (default: top of the plan)")
	verifyCmd := flag.String("vcmd", "", "with -add: the step's verify command (a shell command that exits 0 iff accepted)")
	role := flag.String("role", "", "lane role for dispatch and context (default OVERGO_AUTOMATION_ROLE, then unassigned)")
	flag.Parse()
	if err := run(cli{next: *next, frontier: *frontier, judgeEfficiency: *judgeEfficiency, prompt: *prompt, verify: *verify, status: *status, context: *contextJSON, advance: *advance, add: *add, setverify: *setverify, bindCensus: *bindCensus, pruneDone: *pruneDone, prepareMerge: *prepareMergeFlag, stop: *stop, title: *title, before: *before, verifyCmd: *verifyCmd, role: *role, recordLease: *recordLease, recordLeaseOutcome: *recordLeaseOutcome, grantExploration: *grantExploration, chargeExploration: *chargeExploration, recordExperiment: *recordExperiment, contain: *contain, lane: *lane, localitySchedule: *localitySchedule, leaseReport: *leaseReport, retireLegacyLeases: *retireLegacyLeases, history: *history, admitProposal: *admitProposalFlag, capacity: plan.Resources{CPUThreads: *cpuCapacity, HostRAMGiB: *ramCapacity, VRAMGiB: *vramCapacity}}, flag.Args()); err != nil {
		fmt.Fprintf(os.Stderr, "plan: %v\n", err)
		os.Exit(1)
	}
}

type cli struct {
	next, prompt, verify, status, context, advance, add, setverify, bindCensus, stop bool
	frontier                                                                         bool
	judgeEfficiency                                                                  string
	pruneDone                                                                        bool
	title, before, verifyCmd, role, recordLease, recordLeaseOutcome, contain, lane   string
	prepareMerge                                                                     string
	grantExploration, chargeExploration, recordExperiment                            string
	localitySchedule                                                                 string
	leaseReport                                                                      bool
	retireLegacyLeases                                                               int
	history                                                                          string
	admitProposal                                                                    string
	capacity                                                                         plan.Resources
}

func run(c cli, args []string) error {
	document, err := plan.Load("")
	if err != nil {
		return err
	}
	if err := plan.Validate(document); err != nil {
		return err
	}
	role, err := plan.AutomationRole(c.role)
	if err != nil {
		return err
	}
	if !c.bindCensus {
		if err := plan.ValidateCampaignCensusAuthority(document); err != nil {
			return err
		}
	}
	switch {
	case c.prepareMerge != "":
		return prepareMerge(".", c.prepareMerge, os.Stdout)
	case c.recordLease != "":
		return recordWorkLease(".", c.recordLease, os.Stdout)
	case c.recordLeaseOutcome != "":
		return recordWorkLeaseOutcome(".", c.recordLeaseOutcome, os.Stdout)
	case c.grantExploration != "":
		return recordExplorationGrant(".", c.grantExploration, os.Stdout)
	case c.chargeExploration != "":
		return recordExplorationCharge(".", c.chargeExploration, os.Stdout)
	case c.recordExperiment != "":
		return recordExperimentTransition(".", c.recordExperiment, os.Stdout)
	case c.admitProposal != "":
		return admitProposal(".", c.admitProposal, os.Stdout)
	case c.history != "":
		return printAttemptHistory(c.history, os.Stdout)
	case c.leaseReport:
		return printLeaseReport(".", c.capacity, os.Stdout)
	case c.retireLegacyLeases >= 0:
		return retireLegacyWorkLeases(".", c.retireLegacyLeases, os.Stdout)
	case c.frontier:
		return printReadyFrontier(".", document, os.Stdout)
	case c.judgeEfficiency != "":
		return judgeInteractionEfficiency(c.judgeEfficiency, os.Stdout)
	case c.localitySchedule != "":
		return printLocalitySchedule(".", c.localitySchedule, os.Stdout)
	case c.bindCensus:
		return bindCampaignCensus(".", os.Stdout)
	case c.pruneDone:
		return errors.New("plan: direct pruning is retired; cmd/gate atomically commits and prunes the current row")
	case c.add:
		if len(args) != 1 || strings.TrimSpace(c.title) == "" {
			return errors.New("usage: plan -add <item-id> -title <title> [-before <id>] [-vcmd <verify>]")
		}
		return addItem(".", args[0], c.title, c.before, c.verifyCmd, role)
	case c.setverify:
		if len(args) != 2 || strings.TrimSpace(c.verifyCmd) == "" {
			return errors.New("usage: plan -setverify <item-id> <step-id> -vcmd <cmd>")
		}
		return setStepVerify(".", args[0], args[1], c.verifyCmd, role)
	case c.stop:
		return recordStop(strings.Join(args, " "))
	case c.contain != "":
		return recordControl(c.lane, "containment", c.contain, strings.Join(args, " "))
	case c.advance:
		if len(args) != 2 {
			return errors.New("usage: plan -advance <item-id> <step-id|.>")
		}
		return errors.New("plan: direct advance is retired; commit the current row through cmd/gate")
	case c.status:
		printStatus(document)
		return nil
	case c.context:
		return printAutomationContext(document, c.role, os.Stdout)
	case c.prompt:
		completionAuthority, err := resolveCompletionAuthority(".", document)
		if err != nil {
			return err
		}
		printPrompt(document, role, os.Stdout, completionAuthority)
		return nil
	case c.verify:
		completionAuthority, err := resolveCompletionAuthority(".", document)
		if err != nil {
			return err
		}
		it, st, ok := plan.Current(document, role, completionAuthority)
		if !ok {
			fmt.Println("plan complete: nothing to verify")
			return nil
		}
		return runVerify(it, st)
	case c.next:
		completionAuthority, err := resolveCompletionAuthority(".", document)
		if err != nil {
			return err
		}
		action, open := nextAction(document, role, completionAuthority)
		if !open {
			fmt.Println("plan complete: every item is done")
			return nil
		}
		fmt.Println(action)
		return nil
	default:
		return errors.New("one of -next, -prompt, -verify, -status, -context, -record-lease, -lease-report, -schedule-locality, -advance is required")
	}
}

func resolveCompletionAuthority(root string, document plan.Plan) (plan.CompletionAuthority, error) {
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return plan.CompletionAuthority{}, err
	}
	defer store.Close()
	return plan.ResolveCompletionAuthority(context.Background(), root, "HEAD", document, store)
}

func bindCampaignCensus(root string, output io.Writer) error {
	return withPlanMutation(root, true, func(document plan.Plan) error {
		store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
		if err != nil {
			return err
		}
		defer store.Close()
		id, found, err := artifact.ResolveAlias(context.Background(), store, closurescan.CensusEvidenceAlias)
		if err != nil || !found {
			return errors.Join(err, errors.New("plan: published census evidence is absent"))
		}
		evidence, found, err := closurescan.ReadCensusEvidence(context.Background(), store, id)
		if err != nil || !found || evidence.ID != id {
			return errors.Join(err, errors.New("plan: published census evidence is invalid"))
		}
		document.Census = &id
		if err := plan.ValidateCampaignCensusAuthority(document); err != nil {
			return err
		}
		if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), document); err != nil {
			return err
		}
		fmt.Fprintf(output, "bound campaign census %s source=%s files=%d literals=%d assumptions=%d policy_copies=%d\n",
			id, evidence.Source, evidence.Counts.ProductionFiles, evidence.Counts.InlineLiterals,
			evidence.Counts.AssumptionHints, evidence.Counts.TestPolicyCopies)
		return nil
	})
}

func withPlanMutation(root string, allowCensusRepair bool, mutate func(plan.Plan) error) (err error) {
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
	if !allowCensusRepair {
		if err := plan.ValidateCampaignCensusAuthority(document); err != nil {
			return err
		}
	}
	return mutate(document)
}

func printAutomationContext(document plan.Plan, role string, output io.Writer) error {
	facts, err := collectContextFacts(role)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(filepath.Join(facts.Worktree, "overgodb-store"))
	if err != nil {
		return err
	}
	defer store.Close()
	completions, err := plan.ResolveCompletionAuthority(
		context.Background(), facts.Worktree, facts.Head, document, store,
	)
	if err != nil {
		return err
	}
	facts.EvidenceDebt, facts.Workflow = authoritativeContextEvidence(store, facts.Head)
	context, err := plan.BuildAutomationContext(document, facts, completions)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(output)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(context)
}

func collectContextFacts(role string) (plan.ContextFacts, error) {
	text := func(args ...string) (string, error) {
		out, err := gitOutput(".", args...)
		if err != nil {
			return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
		}
		return strings.TrimSpace(string(out)), nil
	}
	head, err := text("rev-parse", "HEAD")
	if err != nil {
		return plan.ContextFacts{}, err
	}
	branch, err := text("branch", "--show-current")
	if err != nil {
		return plan.ContextFacts{}, err
	}
	if branch == "" {
		branch = "detached"
	}
	worktree, err := text("rev-parse", "--show-toplevel")
	if err != nil {
		return plan.ContextFacts{}, err
	}
	status, err := gitOutput(".", "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		return plan.ContextFacts{}, fmt.Errorf("git status: %w", err)
	}
	dirty, err := repoanalysis.ParseDirtyStatus(status)
	if err != nil {
		return plan.ContextFacts{}, err
	}
	role, err = plan.AutomationRole(role)
	if err != nil {
		return plan.ContextFacts{}, err
	}
	return plan.ContextFacts{
		Head: head, Branch: branch, Worktree: worktree, Role: role, Dirty: dirty,
	}, nil
}

func authoritativeContextEvidence(store *overgodb.Store, head string) (plan.EvidenceDebt, plan.WorkflowContext) {
	const debtSource = "overgodb:overgodb-store"
	const workflowSource = "git:HEAD+overgodb:overgodb-store"
	unavailable := func(reason string) (plan.EvidenceDebt, plan.WorkflowContext) {
		return plan.EvidenceDebt{State: "unknown", Source: debtSource, Reason: reason},
			plan.WorkflowContext{Phase: string(runrecord.ReviewPhaseImplementation), Source: workflowSource, Reason: reason}
	}
	if store == nil {
		return unavailable("store is unavailable")
	}
	ctx := context.Background()
	var lifecycles []runrecord.GateLifecycle
	var candidates []runrecord.ReviewCandidate
	var verdicts []runrecord.ReviewVerdict
	_, err := store.VisitDocuments(ctx, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{
			{Kind: artifact.KindEvidence, MediaType: runrecord.GateLifecycleMediaType, Schema: runrecord.GateLifecycleSchema},
			{Kind: artifact.KindEvidence, MediaType: runrecord.ReviewCandidateMediaType, Schema: runrecord.ReviewCandidateSchema},
			{Kind: artifact.KindEvidence, MediaType: runrecord.ReviewVerdictMediaType, Schema: runrecord.ReviewVerdictSchema},
		}, Order: overgodb.DocumentOldestFirst,
	}, func(view overgodb.DocumentView) error {
		var parseErr error
		switch view.Content.Descriptor.MediaType {
		case runrecord.GateLifecycleMediaType:
			var value runrecord.GateLifecycle
			value, parseErr = runrecord.ParseGateLifecycle(view.Content.Data)
			lifecycles = append(lifecycles, value)
		case runrecord.ReviewCandidateMediaType:
			var value runrecord.ReviewCandidate
			value, parseErr = runrecord.ParseReviewCandidate(view.Content.Data)
			candidates = append(candidates, value)
		case runrecord.ReviewVerdictMediaType:
			var value runrecord.ReviewVerdict
			value, parseErr = runrecord.ParseReviewVerdict(view.Content.Data)
			verdicts = append(verdicts, value)
		}
		return parseErr
	})
	if err != nil {
		return unavailable(err.Error())
	}
	debt, err := runrecord.OutstandingGateDebt(lifecycles)
	debtContext := plan.EvidenceDebt{State: "none_observed", Source: debtSource}
	if err != nil {
		debtContext = plan.EvidenceDebt{State: "unknown", Source: debtSource, Reason: err.Error()}
	} else if len(debt) > 0 {
		debtContext = plan.EvidenceDebt{
			State: "present", Source: debtSource, ResultID: debt[0].ID.String(),
			Reason: fmt.Sprintf("%d prepared gate lifecycle record(s) lack finalization", len(debt)),
		}
	}
	return debtContext, authoritativeReviewPriority(ctx, store, head, candidates, verdicts, workflowSource)
}

func authoritativeReviewPriority(
	ctx context.Context,
	reader artifact.Reader,
	head string,
	candidates []runrecord.ReviewCandidate,
	verdicts []runrecord.ReviewVerdict,
	source string,
) plan.WorkflowContext {
	priority, err := runrecord.DeriveReviewPriority(ctx, reader, head, candidates, verdicts)
	if err != nil {
		return plan.WorkflowContext{Phase: string(runrecord.ReviewPhaseImplementation), Source: source, Reason: err.Error()}
	}
	workflow := plan.WorkflowContext{Phase: string(priority.Phase), Source: source}
	if priority.Candidate.Valid() {
		workflow.CandidateID = priority.Candidate.String()
	}
	if priority.Verdict.Valid() {
		workflow.VerdictID = priority.Verdict.String()
	}
	return workflow
}

// addItem injects a new task as a top-priority item (one step "do"), inserted
// before `before` (or at the top of the plan when empty). This is the mechanical
// "inject a task" operation -- a merge, a fix, or any owner-requested work becomes
// a first-class dispatched/verified/advanced task without hand-editing plan.json.
func addItem(root, id, title, before, verifyCmd, role string) error {
	return withPlanMutation(root, false, func(document plan.Plan) error {
		updated, err := insertItem(document, id, title, before, verifyCmd)
		if err != nil {
			return err
		}
		if role != plan.UnassignedRole {
			for index := range updated.Items {
				if updated.Items[index].ID == id {
					updated.Items[index].Owner = role
					break
				}
			}
		}
		updatedAuthority, err := resolveCompletionAuthority(root, updated)
		if err != nil {
			return err
		}
		if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), updated); err != nil {
			return err
		}
		action, _ := nextAction(updated, role, updatedAuthority)
		fmt.Printf("added item %s (step do); next: %s\n", id, action)
		return nil
	})
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
func setStepVerify(root, itemID, stepID, cmd, role string) error {
	return withPlanMutation(root, false, func(document plan.Plan) error {
		updated, err := assignVerify(document, itemID, stepID, cmd)
		if err != nil {
			return err
		}
		completions, err := resolveCompletionAuthority(root, updated)
		if err != nil {
			return err
		}
		it, st, current := plan.Current(updated, role, completions)
		if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), updated); err != nil {
			return err
		}
		fmt.Printf("set verify for %s/%s: %s\n", itemID, stepID, strings.TrimSpace(cmd))
		if current && it.ID == itemID && st.ID == stepID {
			return runVerify(it, st)
		}
		return nil
	})
}

var validStopReasons = []string{"user-stop", "irreversible", "external-prereq"}

// validateStop returns nil iff reason begins with a legitimate stop tag -- the
// only three reasons that justify ending a turn without plan progress. A
// self-invented "checkpoint" or "should I continue?" is not among them.
func validateStop(reason string) error {
	reason = strings.TrimSpace(reason)
	matched := ""
	for _, v := range validStopReasons {
		if reason == v || strings.HasPrefix(reason, v+":") {
			matched = v
			break
		}
	}
	if matched == "" {
		return fmt.Errorf("invalid stop reason %q -- must begin with one of: user-stop: / irreversible: / external-prereq: <detail>", reason)
	}
	if matched == "external-prereq" {
		// An external prerequisite is something OUTSIDE this process that the
		// work verifiably waits on: a background task, the owner, another
		// lane, or a long-running run. Self-pacing ("fresh context", "next
		// session", "later") is not external and the loop refuses it -- the
		// owner's contract is iterative develop/verify/commit/plan, never a
		// deferral the agent grants itself.
		detail := strings.ToLower(reason)
		for _, selfPacing := range []string{"fresh context", "next session", "context window", "long session", "best started", "another sitting"} {
			if strings.Contains(detail, selfPacing) {
				return fmt.Errorf("stop refused: %q is self-pacing, not an external prerequisite -- continue the plan", selfPacing)
			}
		}
		external := false
		for _, marker := range []string{"background", "running", "owner", "merge", "download", "provision", "lane", "device", "hashing", "gate ", "missing", "absent", "unavailable"} {
			if strings.Contains(detail, marker) {
				external = true
				break
			}
		}
		if !external {
			return errors.New("stop refused: external-prereq detail names nothing external (no background task, owner action, merge, or running work) -- continue the plan")
		}
	}
	return nil
}

// recordStop writes a valid stop marker at the current HEAD; the stop-gate reads
// it to allow a legitimate turn end. Progress on any later turn supersedes it.
func recordStop(reason string) error {
	if err := validateStop(reason); err != nil {
		return err
	}
	head := "unknown"
	if out, err := gitOutput(".", "rev-parse", "HEAD"); err == nil {
		head = strings.TrimSpace(string(out))
	}
	payload := fmt.Sprintf("{\"reason\":%q,\"head\":%q}\n", strings.TrimSpace(reason), head)
	if err := clioptions.WriteOutputFile("docs/plan_stop.json", []byte(payload)); err != nil {
		return err
	}
	fmt.Printf("recorded stop: %s\n", strings.TrimSpace(reason))
	return nil
}

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

// nextAction: the one-line form of the current step.
func nextAction(document plan.Plan, role string, completions plan.CompletionAuthority) (string, bool) {
	it, st, ok := plan.Current(document, role, completions)
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
func printPrompt(document plan.Plan, role string, output io.Writer, completions plan.CompletionAuthority) {
	it, st, ok := plan.Current(document, role, completions)
	if !ok {
		fmt.Fprintln(output, "PLAN COMPLETE: every item is done. Stop and tell the user.")
		return
	}
	verify := st.Verify
	if strings.TrimSpace(verify) == "" {
		verify = "MISSING -- add a runnable step.verify before implementation"
	}
	fmt.Fprintf(output, `TASK %s/%s
%s
VERIFY %s
COMMIT go run ./cmd/gate -plan %s/%s -message-file <msg> -paths <csv>
RULES skill.md; only this task; port-first; park off-scope findings with cmd/finding; gate advances atomically; then rerun plan -prompt.
`, it.ID, st.ID, st.Title, verify, it.ID, st.ID)
}

// runVerify executes the step's verify command; its exit code is the verdict.
func runVerify(it plan.Item, st plan.Step) error {
	if strings.TrimSpace(st.Verify) == "" {
		return fmt.Errorf("no verify defined for %s/%s -- add a runnable step.verify (exits 0 iff accepted) before advancing", it.ID, st.ID)
	}
	fmt.Fprintf(os.Stderr, "plan verify %s/%s: %s\n", it.ID, st.ID, st.Verify)
	class, err := planverify.Execute(context.Background(), ".", st.Verify)
	if err != nil {
		return fmt.Errorf("verify %s/%s: %w -- run the named oracle against its real prerequisite or record an honest stop", it.ID, st.ID, err)
	}
	fmt.Fprintf(os.Stderr, "plan verify %s/%s: PASS verdict=%s\n", it.ID, st.ID, class)
	return nil
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

func recordControl(lane, kind, reason, detail string) error {
	head, err := gitOutput(".", "rev-parse", "HEAD")
	if err != nil {
		return fmt.Errorf("resolve control-event commit: %w", err)
	}
	store, err := overgodb.Open("overgodb-store")
	if err != nil {
		return err
	}
	defer store.Close()
	event, err := plan.RecordControlEvent(context.Background(), store, plan.ControlEvent{
		Kind: kind, Lane: strings.TrimSpace(lane), ReasonCode: reason,
		Detail: strings.TrimSpace(detail), CodeCommit: strings.TrimSpace(string(head)),
	})
	if err != nil {
		return err
	}
	fmt.Printf("recorded %s event %s for %s\n", kind, event.ID, event.Lane)
	return nil
}
