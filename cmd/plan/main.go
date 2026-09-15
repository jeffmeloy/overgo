// plan edits and dispatches the validated campaign in docs/plan.json.
// Dispatch claims work; the gate verifies, publishes and advances it.
// Status and context inspect state without acquiring work.
package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/authoritylock"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/planverify"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
	"overgo/internal/webuilane"
	"overgo/internal/worklease"
)

// Zero is a reviewed lease count; only this sentinel leaves retirement unrequested.
const noLegacyLeaseRetirement = -1

// CLI repository operations resolve from the current worktree directory.
const commandWorktree = "."

func main() {
	edit := flag.String("edit", "", "apply a strict JSON StepEdit file; expected_plan is plan_digest from -context; create/replace never executes verification")
	next := flag.Bool("next", false, "print the top open action")
	frontier := flag.Bool("frontier", false, "print every dispatchable row (the ready frontier) and validate live worktree leases against it")
	judgeEfficiency := flag.String("judge-efficiency", "", "judge an interaction-efficiency claim JSON ({candidate, baseline, tradeoff?}); exit code is the verdict")
	admitProposalFlag := flag.String("admit-proposal", "", "admit one typed steering proposal from a JSON spec into the store and the plan")
	phases := flag.Bool("phases", false, "with -history: show retained phase outcomes and costs; add -json for typed output")
	historyCommit := flag.String("commit", "", "with -history -phases: exact full recorded commit (no revision or prefix expansion)")
	historyResult := flag.String("result", "", "with -history -phases: exact gate-result artifact ID, preserving individual retries")
	history := flag.String("history", "", "print attempt history from the store: aggregates and recent attempts (pass a plan item id, or all)")
	prompt := flag.Bool("prompt", false, "claim an eligible step and print its task: -prompt [-json] [item/step]; requires -worker or OVERGO_AUTOMATION_WORKER")
	verify := flag.Bool("verify", false, "run the top open step's verify command; exit code is pass/fail")
	status := flag.Bool("status", false, "one line per item")
	contextJSON := flag.Bool("context", false, "emit one typed JSON grounding payload for the current automation task")
	recordLease := flag.String("record-lease", "", "record an advisory worktree lease from a JSON file")
	grantExploration := flag.String("grant-exploration", "", "commit an externally-issued exploration budget token from a JSON file")
	chargeExploration := flag.String("charge-exploration", "", "commit one experiment spend against an exploration grant from a JSON file")
	recordLeaseOutcome := flag.String("record-lease-outcome", "", "record measured outcome JSON for an exercised worktree lease")
	recordExperiment := flag.String("record-experiment", "", "commit one experiment lifecycle transition from a JSON spec (state, experiment, evidence, prior)")
	leaseReport := flag.Bool("lease-report", false, "emit active worktree leases and resource/conflict advice as JSON")
	retireLegacyLeases := flag.Int("retire-legacy-leases", noLegacyLeaseRetirement, "retire pre-contract work-lease aliases through reviewed compare-and-set; the value is the reviewed expected count")
	localitySchedule := flag.String("schedule-locality", "", "schedule a JSON worker/artifact request against OvergoDB locations")
	cpuCapacity := flag.Int("cpu-capacity", 0, "with -lease-report: available CPU threads (0 unknown)")
	ramCapacity := flag.Int("ram-capacity-gib", 0, "with -lease-report: available host RAM GiB (0 unknown)")
	vramCapacity := flag.Int("vram-capacity-gib", 0, "with -lease-report: available VRAM GiB (0 unknown)")
	advance := flag.Bool("advance", false, "retired: cmd/gate atomically commits and prunes the current row")
	bindCensus := flag.Bool("bind-census", false, "bind the campaign baseline to closure/census/latest in OvergoDB")
	pruneDone := flag.Bool("prune-done", false, "retired: direct plan pruning is refused")
	add := flag.Bool("add", false, "inject a new top-priority task owned by -role: -add <item-id> -title <t> [-before <id>] [-vcmd <cmd>]")
	move := flag.Bool("move", false, "re-rank an open item: -move <item-id> [-before <id>] (default: top of the plan)")
	retitle := flag.Bool("retitle", false, "re-scope an open item and its single step: -retitle <item-id> -title <t>")
	assign := flag.Bool("assign", false, "record an open item's owning lane: -assign <item-id> -owner <lane>; another lane never dispatches it")
	owner := flag.String("owner", "", "with -assign: the owning lane")
	setLane := flag.String("set-lane", "", "record the lane this plan dispatches for (the unassigned role's role)")
	setverify := flag.Bool("setverify", false, "set an existing step's verify: -setverify <item> <step> -vcmd <cmd> (then runs it; exit code is the verdict)")
	jsonFlag := flag.Bool("json", false, "with -next, -prompt or -history -phases: print typed JSON")
	prepareMergeFlag := flag.String("prepare-merge", "", "snapshot a ref and prepare a gated merge with semantic plan and compatibility regeneration")
	planProjectionFlag := flag.String("plan-projection", "", "with -prepare-merge only: explicit target-plan projection (first-parent-target); empty keeps semantic union")
	stop := flag.Bool("stop", false, "record a scoped stop until explicit resume: -stop <user-stop|irreversible|external-prereq>: <detail>")
	resumeStop := flag.String("resume-stop", "", "explicitly resume the exact stop ID, preserving its owner and scope")
	maintenanceStop := flag.String("maintenance-stop", "", "authorize one supervised task while the exact stop remains active: <item/step> <reason>")
	stopMode := flag.String("stop-mode", "", "stop scope: all (default), interactive or unattended; resume preserves the recorded scope")
	executionMode := flag.String("mode", "", "execution mode: interactive or unattended; default inherited from the driver")
	contain := flag.String("contain", "", "record typed lane containment: -contain <reason-code> -lane <lane> <detail>")
	lane := flag.String("lane", "", "lane affected by -contain")
	_ = flag.String("force", "", "retired with -advance")
	title := flag.String("title", "", "with -add: the task title")
	before := flag.String("before", "", "with -add: insert before this item id (default: top of the plan)")
	verifyCmd := flag.String("vcmd", "", "with -add: the step's verify command (a shell command that exits 0 iff accepted)")
	role := flag.String("role", "", "lane role for dispatch and context (default OVERGO_AUTOMATION_ROLE, then unassigned)")
	worker := flag.String("worker", "", "stable session identity for dispatch claims (default OVERGO_AUTOMATION_WORKER); inherit the same identity in gate subprocesses")
	releaseClaim := flag.String("release-claim", "", "release this worker's exact claim ID; requires -release-reason cancelled or handoff")
	releaseReason := flag.String("release-reason", "", "with -release-claim: cancelled or handoff; retained checks survive release")
	flag.Parse()
	if err := run(cli{edit: *edit, resumeStop: *resumeStop, maintenanceStop: *maintenanceStop, stopMode: *stopMode, mode: *executionMode, json: *jsonFlag, move: *move, retitle: *retitle, assign: *assign, owner: *owner, setLane: *setLane, next: *next, frontier: *frontier, judgeEfficiency: *judgeEfficiency, prompt: *prompt, verify: *verify, status: *status, context: *contextJSON, advance: *advance, add: *add, setverify: *setverify, bindCensus: *bindCensus, pruneDone: *pruneDone, prepareMerge: *prepareMergeFlag, planProjection: *planProjectionFlag, stop: *stop, title: *title, before: *before, verifyCmd: *verifyCmd, role: *role, worker: *worker, releaseClaim: *releaseClaim, releaseReason: *releaseReason, recordLease: *recordLease, recordLeaseOutcome: *recordLeaseOutcome, grantExploration: *grantExploration, chargeExploration: *chargeExploration, recordExperiment: *recordExperiment, contain: *contain, lane: *lane, localitySchedule: *localitySchedule, leaseReport: *leaseReport, retireLegacyLeases: *retireLegacyLeases, history: *history, phases: *phases, historyCommit: *historyCommit, historyResult: *historyResult, admitProposal: *admitProposalFlag, capacity: worklease.Resources{CPUThreads: *cpuCapacity, HostRAMGiB: *ramCapacity, VRAMGiB: *vramCapacity}}, flag.Args()); err != nil {
		fmt.Fprintf(os.Stderr, "plan: %v\n", err)
		os.Exit(1)
	}
}

type cli struct {
	edit                                                                             string
	resumeStop, maintenanceStop, stopMode, mode                                      string
	phases                                                                           bool
	historyCommit, historyResult                                                     string
	releaseClaim, releaseReason                                                      string
	worker                                                                           string
	next, prompt, verify, status, context, advance, add, setverify, bindCensus, stop bool
	move, retitle                                                                    bool
	frontier                                                                         bool
	judgeEfficiency                                                                  string
	pruneDone                                                                        bool
	title, before, verifyCmd, role, recordLease, recordLeaseOutcome, contain, lane   string
	assign                                                                           bool
	owner, setLane                                                                   string
	json                                                                             bool
	prepareMerge                                                                     string
	planProjection                                                                   string
	grantExploration, chargeExploration, recordExperiment                            string
	localitySchedule                                                                 string
	leaseReport                                                                      bool
	retireLegacyLeases                                                               int
	history                                                                          string
	admitProposal                                                                    string
	capacity                                                                         worklease.Resources
}

func run(c cli, args []string) error {
	if c.edit != "" {
		allowed := cli{edit: c.edit, role: c.role, worker: c.worker, json: c.json, retireLegacyLeases: noLegacyLeaseRetirement}
		if c != allowed || len(args) != 0 {
			return errors.New("plan: -edit accepts only its file, -role, -worker and -json")
		}
		return editPlanStep(commandWorktree, c.edit, os.Stdout)
	}

	if c.stop || c.resumeStop != "" || c.maintenanceStop != "" {
		allowed := cli{stop: c.stop, resumeStop: c.resumeStop, maintenanceStop: c.maintenanceStop, stopMode: c.stopMode, role: c.role, worker: c.worker, json: c.json, retireLegacyLeases: noLegacyLeaseRetirement}
		if c != allowed || c.stop && (c.resumeStop != "" || c.maintenanceStop != "") || c.resumeStop != "" && c.maintenanceStop != "" {
			return errors.New("plan: stop control cannot be combined with another operation")
		}
		return recordStopControlCommand(commandWorktree, c, args, os.Stdout)
	}
	if c.stopMode != "" {
		return errors.New("-stop-mode requires a stop control operation")
	}

	if c.releaseClaim != "" || c.releaseReason != "" {
		allowed := cli{releaseClaim: c.releaseClaim, releaseReason: c.releaseReason, worker: c.worker, role: c.role, retireLegacyLeases: noLegacyLeaseRetirement}
		if c != allowed || len(args) != 0 {
			return errors.New("plan: claim release cannot be combined with another operation")
		}
		if c.releaseClaim == "" {
			return errors.New("-release-reason requires -release-claim")
		}
		worker := cmp.Or(strings.TrimSpace(c.worker), strings.TrimSpace(os.Getenv(plan.AutomationWorkerEnvironment)))
		return releaseDispatchClaim(commandWorktree, c.releaseClaim, worker, c.releaseReason, os.Stdout)
	}

	if (c.phases || c.historyCommit != "" || c.historyResult != "") && (c.history == "" || !c.phases) {
		return errors.New("-phases, -commit and -result require -history <item|all> -phases")
	}
	if c.phases {
		query := cli{history: c.history, phases: true, historyCommit: c.historyCommit, historyResult: c.historyResult, json: c.json, retireLegacyLeases: noLegacyLeaseRetirement}
		if len(args) != 0 || c != query {
			return errors.New("-history -phases accepts only -commit, -result and -json; incompatible operation or positional arguments")
		}
		return printGatePhaseHistory(c, os.Stdout)
	}
	if c.next || c.prompt || c.verify {
		return printDispatch(c, args, os.Stdout)
	}
	if c.mode != "" {
		return errors.New("-mode requires -next, -prompt or -verify")
	}
	document, err := loadCommandPlan(c.bindCensus)
	if err != nil {
		return err
	}
	role, err := plan.AutomationRole(c.role)
	if err != nil {
		return err
	}
	switch {
	case c.prepareMerge != "":
		projection, err := plan.ParseMergeProjection(c.planProjection)
		if err != nil {
			return fmt.Errorf("-plan-projection: %w", err)
		}
		return prepareMergeWithProjection(commandWorktree, c.prepareMerge, projection, os.Stdout)
	case c.planProjection != "":
		return errors.New("-plan-projection requires -prepare-merge")
	case c.recordLease != "":
		return recordWorkLease(commandWorktree, c.recordLease, os.Stdout)
	case c.recordLeaseOutcome != "":
		return recordWorkLeaseOutcome(commandWorktree, c.recordLeaseOutcome, os.Stdout)
	case c.grantExploration != "":
		return recordExplorationGrant(commandWorktree, c.grantExploration, os.Stdout)
	case c.chargeExploration != "":
		return recordExplorationCharge(commandWorktree, c.chargeExploration, os.Stdout)
	case c.recordExperiment != "":
		return recordExperimentTransition(commandWorktree, c.recordExperiment, os.Stdout)
	case c.admitProposal != "":
		return admitProposal(commandWorktree, c.admitProposal, os.Stdout)
	case c.history != "":
		return printAttemptHistory(c.history, os.Stdout)
	case c.leaseReport:
		return printLeaseReport(commandWorktree, c.capacity, os.Stdout)
	case c.retireLegacyLeases >= 0:
		return retireLegacyWorkLeases(commandWorktree, c.retireLegacyLeases, os.Stdout)
	case c.frontier:
		return printReadyFrontier(commandWorktree, document, os.Stdout)
	case c.judgeEfficiency != "":
		return judgeInteractionEfficiency(c.judgeEfficiency, os.Stdout)
	case c.localitySchedule != "":
		return printLocalitySchedule(commandWorktree, c.localitySchedule, os.Stdout)
	case c.bindCensus:
		return bindCampaignCensus(commandWorktree, os.Stdout)
	case c.pruneDone:
		return errors.New("plan: direct pruning is retired; cmd/gate atomically commits and prunes the current row")
	case c.add:
		if len(args) != 1 || strings.TrimSpace(c.title) == "" {
			return errors.New("usage: plan -add <item-id> -title <title> [-before <id>] [-vcmd <verify>]")
		}
		return addItem(commandWorktree, args[0], c.title, c.before, c.verifyCmd, role)
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
	case c.assign:
		if len(args) != 1 || strings.TrimSpace(c.owner) == "" {
			return errors.New("usage: plan -assign <item-id> -owner <lane>")
		}
		return mutatePlan(commandWorktree, role, "assigned item "+args[0]+" to "+strings.TrimSpace(c.owner), func(document plan.Plan) (plan.Plan, error) { return assignOwner(document, args[0], c.owner) })
	case c.setLane != "":
		return mutatePlan(commandWorktree, role, "set lane "+strings.TrimSpace(c.setLane), func(document plan.Plan) (plan.Plan, error) { return setPlanLane(document, c.setLane) })
	case c.setverify:
		if len(args) != 2 || strings.TrimSpace(c.verifyCmd) == "" {
			return errors.New("usage: plan -setverify <item-id> <step-id> -vcmd <cmd>")
		}
		if browserVerifyOutsideLane(c.verifyCmd) {
			return errors.New("plan: a browser test is evidence only through cmd/webui-lane (outside it the test skips); name the lane with -run and -require in the verify")
		}
		return setStepVerify(commandWorktree, args[0], args[1], c.verifyCmd, role)
	case c.contain != "":
		return recordControl(c.lane, "containment", c.contain, strings.Join(args, commandWordSeparator))
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
		if err := savePlanMutation(root, document); err != nil {
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
	stop, err := plan.ReadStop(context.Background(), store, facts.Worktree, facts.Head, "")
	if err != nil {
		return err
	}
	context, err := plan.BuildAutomationContext(document, facts, completions)
	if err != nil {
		return err
	}
	if stop.State != "absent" {
		context.Stop = &stop
		if stop.Blocked {
			context.PlanState = "stopped"
		}
	}
	encoder := json.NewEncoder(output)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(context)
}

func collectContextFacts(role string) (plan.ContextFacts, error) {
	text := func(args ...string) (string, error) {
		out, err := gitOutput(commandWorktree, args...)
		if err != nil {
			return "", fmt.Errorf("git %s: %w", strings.Join(args, commandWordSeparator), err)
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
	branch = cmp.Or(branch, "detached")
	worktree, err := text("rev-parse", "--show-toplevel")
	if err != nil {
		return plan.ContextFacts{}, err
	}
	status, err := gitOutput(commandWorktree, "status", "--porcelain=v1", "-z", "--untracked-files=all")
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
		if role != worklease.UnassignedRole {
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
		if err := savePlanMutation(root, updated); err != nil {
			return err
		}
		action, _ := nextAction(updated, role, updatedAuthority)
		fmt.Printf("added item %s (step do); next: %s\n", id, action)
		return nil
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
	return withPlanMutation(root, false, func(document plan.Plan) error {
		updated, err := mutation(document)
		if err != nil {
			return err
		}
		updatedAuthority, err := resolveCompletionAuthority(root, updated)
		if err != nil {
			return err
		}
		if err := savePlanMutation(root, updated); err != nil {
			return err
		}
		action, _ := nextAction(updated, role, updatedAuthority)
		fmt.Printf("%s; next: %s\n", what, action)
		return nil
	})
}

// assignOwner is the pure core of -assign: the named open item takes the owner. No I/O.
func assignOwner(document plan.Plan, id, owner string) (plan.Plan, error) {
	owner = strings.TrimSpace(owner)
	if owner == "" {
		return plan.Plan{}, errors.New("assign: the owner is empty")
	}
	index := slices.IndexFunc(document.Items, func(item plan.Item) bool { return item.ID == id })
	if index < 0 {
		return plan.Plan{}, fmt.Errorf("assign: no item %q", id)
	}
	document.Items = slices.Clone(document.Items)
	document.Items[index].Owner = owner
	return document, nil
}

// setPlanLane is the pure core of -set-lane. No I/O.
func setPlanLane(document plan.Plan, lane string) (plan.Plan, error) {
	if lane = strings.TrimSpace(lane); lane == "" {
		return plan.Plan{}, errors.New("set-lane: the lane is empty")
	}
	document.Lane = lane
	return document, nil
}

// rescopeItem is the pure core of retitleItem. No I/O.
func rescopeItem(document plan.Plan, id, title string) (plan.Plan, error) {
	title = strings.TrimSpace(title)
	if title == "" {
		return plan.Plan{}, errors.New("retitle: the title is empty")
	}
	for index := range document.Items {
		if document.Items[index].ID != id {
			continue
		}
		document.Items[index].Title = title
		if steps := document.Items[index].Steps; len(steps) == 1 && steps[0].ID == "do" {
			document.Items[index].Steps[0].Title = title
		}
		return document, nil
	}
	return plan.Plan{}, fmt.Errorf("item %q: no such item", id)
}

// relocateItem is the pure core of moveItem: returns a plan with the item
// removed from its position and reinserted before `before` (or at the top
// when empty). Moving an item before itself is the identity. No I/O.
func relocateItem(document plan.Plan, id, before string) (plan.Plan, error) {
	if id == before {
		return document, nil
	}
	from := -1
	for index, it := range document.Items {
		if it.ID == id {
			from = index
			break
		}
	}
	if from < 0 {
		return plan.Plan{}, fmt.Errorf("item %q: no such item", id)
	}
	item := document.Items[from]
	rest := make([]plan.Item, 0, len(document.Items))
	rest = append(rest, document.Items[:from]...)
	rest = append(rest, document.Items[from+1:]...)
	pos := 0
	if before != "" {
		pos = -1
		for index, it := range rest {
			if it.ID == before {
				pos = index
				break
			}
		}
		if pos < 0 {
			return plan.Plan{}, fmt.Errorf("-before %q: no such item", before)
		}
	}
	out := make([]plan.Item, 0, len(document.Items))
	out = append(out, rest[:pos]...)
	out = append(out, item)
	out = append(out, rest[pos:]...)
	document.Items = out
	return document, nil
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
func nextAction(document plan.Plan, role string, completions plan.CompletionAuthority) (string, bool) {
	dispatch := plan.DispatchOf(document, role, completions)
	if dispatch.Complete {
		return "", false
	}
	return dispatch.Line, true
}

// printPrompt emits the self-contained, non-negotiable task for the current
// step. The loop feeds THIS to the agent; the agent does not author it.
func printPrompt(document plan.Plan, role string, output io.Writer, completions plan.CompletionAuthority) {
	it, st, ok := plan.Current(document, role, completions)
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
	fmt.Fprintf(output, `TASK %s/%s
%s
VERIFY %s
COMMIT go run ./cmd/gate -plan %s/%s -message-file <msg> -paths <csv>
RULES skill.md; only this task; port-first; park off-scope findings with cmd/finding; gate advances atomically; then rerun plan -prompt.
`, it.ID, st.ID, st.Title, verify, it.ID, st.ID)
	if st.VerificationBatch != nil {
		for _, checkpoint := range st.VerificationBatch.Checkpoints {
			fmt.Fprintf(output, "BATCH ACCEPTANCE %s: %s\n", checkpoint.ID, checkpoint.Verify)
		}
		fmt.Fprintln(output, "BATCH declarations add required acceptance; full gating and parent completion remain mandatory.")
	}
}

// browserVerifyOutsideLane: a verify naming a browser acceptance test without the lane runner that makes it run.
func browserVerifyOutsideLane(command string) bool {
	return strings.Contains(command, webuilane.BrowserTestPrefix) && !strings.Contains(command, "cmd/webui-lane")
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
	head, err := gitOutput(commandWorktree, "rev-parse", "HEAD")
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

// All command paths use the same document and campaign validation sequence.
func loadCommandPlan(bindingCensus bool) (plan.Plan, error) {
	document, err := plan.Load("")
	if err != nil {
		return document, err
	}
	if err := plan.Validate(document); err != nil {
		return document, err
	}
	if !bindingCensus {
		if err := plan.ValidateCampaignCensusAuthority(document); err != nil {
			return document, err
		}
	}
	return document, nil
}
