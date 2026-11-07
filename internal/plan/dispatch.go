package plan

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/worklease"
)

// Dispatch is the current row as data: the harness reads its fields and
// Line is the one-line prose rendered from the same fields.
type Dispatch struct {
	Stop      *StopStatus `json:"stop,omitempty"`
	Complete  bool        `json:"complete"`
	Item      string      `json:"item,omitzero"`
	Step      string      `json:"step,omitzero"`
	ItemTitle string      `json:"item_title,omitzero"`
	StepTitle string      `json:"step_title,omitzero"`
	Verify    string      `json:"verify,omitzero"`
	Line      string      `json:"line"`
	Waiting   string      `json:"waiting,omitzero"`
	// Decided names each unanswered decision the row goes on under, at its recommended option (byRecommendation).
	Decided []string `json:"decided,omitempty"`
}

// Shared Git revision query used by dispatch.
const gitRevisionCommand = "rev-parse"

// Git names the current worktree revision through HEAD.
const gitHeadRevision = "HEAD"

// DispatchRequest names whose row is resolved: the role and worker a stop
// is scoped by, the execution mode, a maintenance grant, and optionally the
// one row the caller asks for.
type DispatchRequest struct {
	Mode        string
	Maintenance string
	Role        string
	Worker      string
	Reference   string
}

// AutomationWorkerEnvironment is inherited by a driver and its gate subprocesses.
const AutomationWorkerEnvironment = "OVERGO_AUTOMATION_WORKER"

func dispatchWorker(explicit string) (string, error) {
	worker := cmp.Or(strings.TrimSpace(explicit), strings.TrimSpace(os.Getenv(AutomationWorkerEnvironment)))
	if worker != "" && (!worklease.ValidAutomationText(worker) || worker == worklease.UnassignedRole) {
		return "", errors.New("plan: worker must be a stable distinct session identity")
	}
	return worker, nil
}

// DispatchOf resolves the current row for role under a resolved authority.
func DispatchOf(document Plan, role string) Dispatch {
	item, step, ok := Current(document, role)
	if !ok {
		return finishDispatch(Dispatch{Complete: true})
	}
	return stepDispatch(item, step)
}

func stepDispatch(item Item, step Step) Dispatch {
	return finishDispatch(Dispatch{Item: item.ID, Step: step.ID, ItemTitle: item.Title, StepTitle: step.Title, Verify: step.Verify})
}

func finishDispatch(dispatch Dispatch) Dispatch {
	dispatch.Line = dispatchLine(dispatch)
	return dispatch
}

func dispatchLine(dispatch Dispatch) string {
	switch {
	case dispatch.Waiting != "":
		return "plan waiting: " + dispatch.Waiting
	case dispatch.Complete:
		return "plan complete: every item is done"
	case dispatch.Step == ".":
		return fmt.Sprintf("%s: %s -- open the rung (define its steps)", dispatch.Item, dispatch.ItemTitle) + decidedNote(dispatch)
	default:
		return fmt.Sprintf("%s / %s: %s -- %s", dispatch.Item, dispatch.Step, dispatch.ItemTitle, dispatch.StepTitle) + decidedNote(dispatch)
	}
}

// decidedNote says which unanswered decisions a row goes on under, at their recommended options.
func decidedNote(dispatch Dispatch) string {
	if len(dispatch.Decided) == 0 {
		return ""
	}
	return " (unanswered, taken at the recommended option: " + strings.Join(dispatch.Decided, "; ") + ")"
}

// ResolveDispatch resolves the current row for the repository at root and
// role: the first ready row in prerequisite order, unless a scoped stop
// holds the loop. It reads the plan and the stop file and writes neither.
func ResolveDispatch(ctx context.Context, root string, request DispatchRequest) (Dispatch, error) {
	role, err := AutomationRole(request.Role)
	if err != nil {
		return Dispatch{}, err
	}
	worker, err := dispatchWorker(request.Worker)
	if err != nil {
		return Dispatch{}, err
	}
	repository, head, err := resolveRevision(ctx, root, gitHeadRevision)
	if err != nil {
		return Dispatch{}, err
	}
	mode, err := ExecutionMode(request.Mode)
	if err != nil {
		return Dispatch{}, err
	}
	stop, err := ReadStop(ctx, repository, head, mode)
	if err != nil {
		return Dispatch{}, err
	}
	grant := cmp.Or(request.Maintenance, strings.TrimSpace(os.Getenv(AutomationMaintenanceEnvironment)))
	if stop.Blocked && grant == "" {
		return finishDispatch(Dispatch{Stop: &stop, Waiting: stop.String()}), nil
	}
	document, err := Load(filepath.Join(repository, filepath.FromSlash(Path)))
	if err != nil {
		return Dispatch{}, err
	}
	if document.parked, err = readParked(repository, head); err != nil {
		return Dispatch{}, err
	}
	dispatch, err := selectDispatch(document, role, request.Reference)
	if err != nil {
		return Dispatch{}, err
	}
	if stop.State != stopAbsent {
		dispatch.Stop = &stop
	}
	if stopErr := stop.RequireExecution(role, worker, mode, dispatch.Item+"/"+dispatch.Step, grant); stopErr != nil {
		dispatch.Waiting = stopErr.Error()
		dispatch = finishDispatch(dispatch)
	}
	return dispatch, nil
}

func dispatchStep(document Plan, ref string) (Item, Step, bool) {
	for _, item := range document.Items {
		for _, step := range item.Steps {
			if item.ID+"/"+step.ID == ref {
				return item, step, true
			}
		}
	}
	return Item{}, Step{}, false
}

// Visit prerequisites before the blocked objective's later siblings; retain
// document order otherwise. This changes scheduling, never dependency evidence.
func dispatchPriority(document Plan, role string) ([]Ref, error) {
	frontier, err := ReadyFrontier(document)
	if err != nil {
		return nil, err
	}
	ready := make(map[string]bool, len(frontier))
	for _, ref := range frontier {
		ready[ref.String()] = true
	}
	role = normalizedRole(role)
	if role == worklease.UnassignedRole && document.Lane != "" {
		role = document.Lane
	}
	awaiting := awaitingDecision(document)
	eligible := func(item Item) bool {
		return !awaiting[item.ID] && (item.Owner == "" || role != worklease.UnassignedRole && item.Owner == role)
	}
	visited := map[string]bool{}
	var ordered []Ref
	var visit func(Item, Step)
	visit = func(item Item, step Step) {
		ref := Ref{Item: item.ID, Step: step.ID}
		if visited[ref.String()] || item.Status != StatusOpen || step.Status != StatusOpen {
			return
		}
		visited[ref.String()] = true
		if ready[ref.String()] {
			if eligible(item) && !slices.Contains(document.parked, ref.String()) {
				ordered = append(ordered, ref)
			}
			return
		}
		for _, dependency := range step.DependsOn {
			if parent, prerequisite, found := dispatchStep(document, dependency); found {
				visit(parent, prerequisite)
			}
		}
	}
	owners := []string{""}
	if role != worklease.UnassignedRole {
		owners = []string{role, ""}
	}
	for _, owner := range owners {
		for _, item := range document.Items {
			if item.Owner == owner {
				for _, step := range item.Steps {
					visit(item, step)
				}
			}
		}
	}
	return ordered, nil
}

// selectDispatch picks the first ready row in prerequisite order, or the
// requested row when it is ready.
func selectDispatch(document Plan, role, reference string) (Dispatch, error) {
	ordered, err := dispatchPriority(document, role)
	if err != nil {
		return Dispatch{}, err
	}
	for _, ref := range ordered {
		if reference != "" && ref.String() != reference {
			continue
		}
		item, step, _ := dispatchStep(document, ref.String())
		return stepDispatch(item, step), nil
	}
	awaiting := awaitingDecision(document)
	if item, step, open := Current(document, role); open && !awaiting[item.ID] && reference == "" &&
		!slices.Contains(document.parked, item.ID+"/"+step.ID) {
		// Preserve the empty-rung dispatch until the rung is given steps.
		return DispatchOf(document, role), nil
	}
	if decided, taken := byRecommendation(document); taken && slices.ContainsFunc(document.Items, func(item Item) bool {
		return item.Status == StatusOpen && awaiting[item.ID]
	}) {
		dispatch, err := selectDispatch(decided, role, reference)
		if err == nil && dispatch.Item != "" {
			dispatch.Decided = decidedFor(document, dispatch.Item)
			dispatch.Line = dispatchLine(dispatch)
		}
		return dispatch, err
	}
	if reference != "" {
		return finishDispatch(Dispatch{Waiting: "requested row is not ready for this role"}), nil
	}
	var held []string
	for _, item := range document.Items {
		if decision, holding := holdingDecision(document, item.ID); holding && item.Status == StatusOpen {
			held = append(held, item.ID+" is held by decision "+decision.ID+": "+decision.holds())
		}
	}
	held = append(held, parkedNote(document)...)
	if len(held) != 0 {
		return finishDispatch(Dispatch{Waiting: strings.Join(held, "; ")}), nil
	}
	for _, item := range document.Items {
		if item.Status == StatusOpen {
			return finishDispatch(Dispatch{Waiting: "remaining rows require prerequisite evidence or an eligible role"}), nil
		}
	}
	return finishDispatch(Dispatch{Complete: true}), nil
}

// AdmitLanding refuses refs, landed together on document in the worktree at root, that dispatch would not give the
// request's role: any step while a recorded stop holds its mode, a step that waits on a prerequisite or a decision,
// another lane's row, and, for a worker, any step but the one dispatch names, the batch's earlier steps having left.
func AdmitLanding(ctx context.Context, root string, document Plan, request DispatchRequest, refs []Ref) error {
	role, err := AutomationRole(request.Role)
	if err != nil {
		return err
	}
	worker, err := dispatchWorker(request.Worker)
	if err != nil {
		return err
	}
	mode, err := ExecutionMode(request.Mode)
	if err != nil {
		return err
	}
	stop, err := ReadStop(ctx, root, "", mode)
	if err != nil {
		return err
	}
	grant := cmp.Or(request.Maintenance, strings.TrimSpace(os.Getenv(AutomationMaintenanceEnvironment)))
	for _, ref := range refs {
		if err := stop.RequireExecution(role, worker, mode, ref.String(), grant); err != nil {
			return fmt.Errorf("plan: a recorded stop holds %s: %w", ref, err)
		}
		if err := admits(document, role, worker != "", ref); err != nil {
			return err
		}
		if document, err = advancePlan(document, ref.Item, ref.Step, ref.Step == "."); err != nil {
			return err
		}
	}
	return nil
}

// admits refuses ref unless dispatch gives it to role: for a worker, or a rung with no steps, the one step dispatch
// names, and otherwise any step it would give.
func admits(document Plan, role string, worker bool, ref Ref) error {
	reference := ref.String()
	if worker || ref.Step == "." {
		reference = ""
	}
	dispatch, err := selectDispatch(document, role, reference)
	switch {
	case err != nil:
		return err
	case dispatch.Item == ref.Item && dispatch.Step == ref.Step:
		return nil
	case worker:
		return fmt.Errorf("plan: a worker lands only the step dispatch gives it (%s), never %s", dispatch.Line, ref)
	}
	return fmt.Errorf("plan: dispatch would not give %s to %s: %s", ref, role, dispatch.Line)
}

// byRecommendation is the plan with each unanswered decision taken at its recommendation, and whether it took any: once
// no row is ready but those a question blocks, they go on under the recommended option, and an answer given before then
// is the one they go on under. The loop never waits on a question, except one that changes a standing rule.
func byRecommendation(document Plan) (Plan, bool) {
	decided := slices.Clone(document.Decisions)
	taken := false
	for at := range decided {
		if decided[at].Answer == "" && !decided[at].waitsForOwner() {
			decided[at].Answer = decided[at].Recommendation
			taken = true
		}
	}
	document.Decisions = decided
	return document, taken
}

// decidedFor names the unanswered decisions an item goes on under.
func decidedFor(document Plan, item string) []string {
	var taken []string
	for _, decision := range document.Decisions {
		if decision.Answer == "" && !decision.waitsForOwner() && slices.Contains(decision.Blocks, item) {
			taken = append(taken, decision.ID+": "+decision.Recommendation)
		}
	}
	return taken
}

// awaitingDecision are the items a decision keeps from running as they stand: its rows were written for the recommended
// option. Dispatch moves past them while another row is ready (byRecommendation).
func awaitingDecision(document Plan) map[string]bool {
	blocked := map[string]bool{}
	for _, decision := range document.Decisions {
		if decision.Answer != decision.Recommendation {
			for _, item := range decision.Blocks {
				blocked[item] = true
			}
		}
	}
	return blocked
}

// holdingDecision is the decision an item cannot run under: one that changes a standing rule and has no answer, or one
// answered another way than the row was written for, until the row is rewritten or retired.
func holdingDecision(document Plan, item string) (Decision, bool) {
	at := slices.IndexFunc(document.Decisions, func(decision Decision) bool {
		return slices.Contains(decision.Blocks, item) && decision.Answer != decision.Recommendation &&
			(decision.Answer != "" || decision.waitsForOwner())
	})
	if at < 0 {
		return Decision{}, false
	}
	return document.Decisions[at], true
}

// parkedNote says which rows wait for a commit.
func parkedNote(document Plan) []string {
	if len(document.parked) == 0 {
		return nil
	}
	return []string{"parked until a commit: " + strings.Join(document.parked, ", ")}
}

// holds says why a holding decision holds its rows.
func (decision Decision) holds() string {
	switch {
	case decision.Answer == "" && decision.OwnersAct:
		return "its answer is an act only the owner takes, so it waits for the owner's answer while other rows run"
	case decision.Answer == "":
		return "it changes a standing rule, so it waits for the owner's answer"
	}
	return "answered " + strconv.Quote(decision.Answer) + ", and the row was written for another answer; rewrite or retire it"
}
