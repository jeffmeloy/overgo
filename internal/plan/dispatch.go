package plan

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
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
		return fmt.Sprintf("%s: %s -- open the rung (define its steps)", dispatch.Item, dispatch.ItemTitle)
	default:
		return fmt.Sprintf("%s / %s: %s -- %s", dispatch.Item, dispatch.Step, dispatch.ItemTitle, dispatch.StepTitle)
	}
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
			if eligible(item) {
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
	if reference != "" {
		return finishDispatch(Dispatch{Waiting: "requested row is not ready for this role"}), nil
	}
	awaiting := awaitingDecision(document)
	if item, _, open := Current(document, role); open && !awaiting[item.ID] {
		// Preserve the empty-rung dispatch until the rung is given steps.
		return DispatchOf(document, role), nil
	}
	for _, item := range document.Items {
		if item.Status == StatusOpen && awaiting[item.ID] {
			return finishDispatch(Dispatch{Waiting: "remaining rows wait on owner decisions; see plan -decisions"}), nil
		}
	}
	for _, item := range document.Items {
		if item.Status == StatusOpen {
			return finishDispatch(Dispatch{Waiting: "remaining rows require prerequisite evidence or an eligible role"}), nil
		}
	}
	return finishDispatch(Dispatch{Complete: true}), nil
}

// awaitingDecision are the items an unanswered owner decision blocks:
// dispatch moves past them until the owner answers.
func awaitingDecision(document Plan) map[string]bool {
	blocked := map[string]bool{}
	for _, decision := range document.Decisions {
		if decision.Answer == "" {
			for _, item := range decision.Blocks {
				blocked[item] = true
			}
		}
	}
	return blocked
}
