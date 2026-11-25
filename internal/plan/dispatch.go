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

	"overgo/internal/processcontrol"
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
	// Findings, of an empty plan, are the open P1 and P2 findings no row names, to be filed as rows before the
	// measurement run (FindingsToFile).
	Findings []string `json:"findings,omitempty"`
	// Answer names each unanswered decision that changes a standing rule and holds a waiting row: the lead answers it
	// now, and a session's turn does not end on it (doctrine-answer).
	Answer []string `json:"answer,omitempty"`
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

// DispatchOf is the step the plan alone dispatches, as -next selects it: the first ready by prerequisite, decision and
// parked row, not the first open, which a decision may hold. What the repository records beside the plan, a stop, the
// card's holder and the findings register, is ResolveDispatch's.
func DispatchOf(document Plan) Dispatch {
	dispatch, err := selectDispatch(document, "")
	if err != nil {
		return finishDispatch(Dispatch{Waiting: err.Error()})
	}
	return dispatch
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
	case dispatch.Complete && len(dispatch.Findings) != 0:
		return "plan complete: every item is done; before the measurement run, file as rows the open findings no row names: " +
			strings.Join(dispatch.Findings, "; ")
	case dispatch.Complete:
		return "plan complete: every item is done"
	case dispatch.Step == ".":
		return fmt.Sprintf("%s: %s -- open the rung (define its steps)", dispatch.Item, dispatch.ItemTitle) + decidedNote(dispatch)
	default:
		return firstP1(dispatch) + fmt.Sprintf("%s / %s: %s -- %s", dispatch.Item, dispatch.Step, dispatch.ItemTitle, dispatch.StepTitle) +
			decidedNote(dispatch) + findingsNote(dispatch)
	}
}

// firstP1 puts an unanswered P1 finding no row names ahead of the step, until a row names it or the triage answers it.
func firstP1(dispatch Dispatch) string {
	for _, finding := range dispatch.Findings {
		if strings.Contains(finding, " (P1): ") {
			return "first file as a row the P1 finding no row names, " + finding + "; then "
		}
	}
	return ""
}

// findingsNote names by id the open findings no row names, which a step is dispatched beside.
func findingsNote(dispatch Dispatch) string {
	ids := make([]string, 0, len(dispatch.Findings))
	for _, finding := range dispatch.Findings {
		id, _, _ := strings.Cut(finding, " ")
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return ""
	}
	return " (findings no row names: " + strings.Join(ids, ", ") + ")"
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
	// A stop on an outside need lifts itself once the check it names passes.
	if stop, _, err = clearOutsideStop(ctx, repository, head, mode, stop); err != nil {
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
	if document, err = besideThePlan(document, repository, head, processcontrol.CardTurnPrefix); err != nil {
		return Dispatch{}, err
	}
	if other := otherLane(document, role); other != "" {
		return finishDispatch(Dispatch{Waiting: other}), nil
	}
	dispatch, err := selectDispatch(document, request.Reference)
	if err != nil {
		return Dispatch{}, err
	}
	// Every dispatch reads the register's open P1 and P2 findings no row names: an empty plan falls through to them
	// before the measurement run, and a step is dispatched with them, an unanswered P1 ahead of it. Read only for an
	// empty plan, none was dispatched from 11 November, while the plan held rows (owner, 2026-11-13).
	if register, readErr := os.ReadFile(filepath.Join(repository, filepath.FromSlash(FindingsRegister))); readErr == nil {
		if dispatch.Findings, err = FindingsToFile(document, register); err != nil {
			return Dispatch{}, err
		}
		dispatch.Line = dispatchLine(dispatch)
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
// document order otherwise. A step that waits for the card another holds
// (cardWaits) comes after every one that does not, its card work queued at the
// card's turn. This changes scheduling, never dependency evidence.
func dispatchPriority(document Plan) ([]Ref, error) {
	frontier, err := ReadyFrontier(document)
	if err != nil {
		return nil, err
	}
	ready := make(map[string]bool, len(frontier))
	for _, ref := range frontier {
		ready[ref.String()] = true
	}
	awaiting := awaitingDecision(document)
	visited := map[string]bool{}
	var ordered, carded []Ref
	var visit func(Item, Step)
	visit = func(item Item, step Step) {
		ref := Ref{Item: item.ID, Step: step.ID}
		if visited[ref.String()] {
			return
		}
		visited[ref.String()] = true
		if ready[ref.String()] {
			switch {
			case awaiting[item.ID] || slices.Contains(document.parked, ref.String()):
			case document.cardWaits(step):
				carded = append(carded, ref)
			default:
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
	for _, item := range document.Items {
		for _, step := range item.Steps {
			visit(item, step)
		}
	}
	return append(ordered, carded...), nil
}

// selectDispatch picks the first ready row in prerequisite order, or the
// requested row when it is ready.
func selectDispatch(document Plan, reference string) (Dispatch, error) {
	ordered, err := dispatchPriority(document)
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
	if item, step, open := Current(document); open && !awaiting[item.ID] && reference == "" &&
		!slices.Contains(document.parked, item.ID+"/"+step.ID) {
		// Preserve the empty-rung dispatch until the rung is given steps.
		return stepDispatch(item, step), nil
	}
	if decided, taken := byRecommendation(document); taken && slices.ContainsFunc(document.Items, func(item Item) bool {
		return awaiting[item.ID]
	}) {
		dispatch, err := selectDispatch(decided, reference)
		if err == nil && dispatch.Item != "" {
			dispatch.Decided = decidedFor(document, dispatch.Item)
			dispatch.Line = dispatchLine(dispatch)
		}
		return dispatch, err
	}
	if reference != "" {
		return finishDispatch(Dispatch{Waiting: "requested row is not ready"}), nil
	}
	var held, answer []string
	for _, item := range document.Items {
		if decision, holding := holdingDecision(document, item.ID); holding {
			held = append(held, item.ID+" is held by decision "+decision.ID+": "+decision.holds())
			if decision.Answer == "" && decision.Doctrine && !decision.OwnersAct && !slices.Contains(answer, decision.ID) {
				answer = append(answer, decision.ID)
			}
		}
	}
	held = append(held, parkedNote(document)...)
	if len(held) != 0 {
		return finishDispatch(Dispatch{Waiting: strings.Join(held, "; "), Answer: answer}), nil
	}
	if len(document.Items) != 0 {
		return finishDispatch(Dispatch{Waiting: "remaining rows require prerequisite evidence"}), nil
	}
	return finishDispatch(Dispatch{Complete: true}), nil
}

// besideThePlan is document with what the repository records beside it, which dispatch and admission both read: the
// rows parked at head, and the card's holder, the measurement run or another process holding a turn named under turn.
// Admission read neither, so while the readings campaign held the card on 2026-11-16 -next gave the step that needs
// none and the gate refused it for the card step (admit-card/do).
func besideThePlan(document Plan, repository, head, turn string) (Plan, error) {
	var err error
	if document.parked, err = readParked(repository, head); err != nil {
		return Plan{}, err
	}
	run, recorded := ReadMeasurementRun(repository)
	if document.cardHolder, err = cardHolder(run, recorded, turn); err != nil {
		return Plan{}, err
	}
	return document, nil
}

// AdmitLanding refuses refs, landed together on document in the worktree at root, that dispatch would not give the
// request's role: any step while a recorded stop holds its mode, a step that waits on a prerequisite or a decision,
// another lane's row, and, for a worker, any step but the one dispatch names, the batch's earlier steps having left.
// It reads what dispatch reads beside the plan (besideThePlan).
func AdmitLanding(ctx context.Context, root string, document Plan, request DispatchRequest, refs []Ref) error {
	return admitLanding(ctx, root, document, request, refs, processcontrol.CardTurnPrefix)
}

// admitLanding is AdmitLanding with the card's turns read under turn.
func admitLanding(ctx context.Context, root string, document Plan, request DispatchRequest, refs []Ref, turn string) error {
	repository, head, err := resolveRevision(ctx, root, gitHeadRevision)
	if err != nil {
		return err
	}
	if document, err = besideThePlan(document, repository, head, turn); err != nil {
		return err
	}
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
// names with the card held or free, and otherwise any step it would give.
func admits(document Plan, role string, worker bool, ref Ref) error {
	if other := otherLane(document, role); other != "" {
		return errors.New("plan: " + other)
	}
	reference := ref.String()
	if worker || ref.Step == "." {
		reference = ""
	}
	dispatch, err := selectDispatch(document, reference)
	switch {
	case err != nil:
		return err
	case dispatch.Item == ref.Item && dispatch.Step == ref.Step:
		return nil
	case worker && givenAtOtherReading(document, reference, ref):
		return nil
	case worker:
		return fmt.Errorf("plan: a worker lands only the step dispatch gives it (%s), never %s", dispatch.Line, ref)
	}
	return fmt.Errorf("plan: dispatch would not give %s to %s: %s", ref, role, dispatch.Line)
}

// givenAtOtherReading says dispatch gives ref with the card read the other way, free where it is held and held where
// it is free (dispatch-steady/do). Who holds the card orders what a worker is offered, so it waits less, and does not
// make a step wrong to land: a card step's landing waits for its turn, which the lanes yield to a commit. The lanes
// runner takes the card's turn package by package, so the reading flips: on 2026-11-24 a step whose check is on the
// card was refused while the runner held the turn, and the step dispatch had then given was refused minutes later
// while it held none.
func givenAtOtherReading(document Plan, reference string, ref Ref) bool {
	other := document
	if other.cardHolder == "" {
		other.cardHolder = "another process"
	} else {
		other.cardHolder = ""
	}
	dispatch, err := selectDispatch(other, reference)
	return err == nil && dispatch.Item == ref.Item && dispatch.Step == ref.Step
}

// otherLane says why role takes no row of the plan, which is its lane's alone: a role named that is another lane's
// (single-source/step-fields, where every row's owner was the plan's lane); empty when role takes its rows.
func otherLane(document Plan, role string) string {
	if role = normalizedRole(role); document.Lane == "" || role == worklease.UnassignedRole || role == document.Lane {
		return ""
	}
	return "the plan is lane " + document.Lane + "'s, and gives role " + role + " no row"
}

// byRecommendation is the plan with each unanswered decision taken at its recommendation, and whether it took any: once
// no row is ready but those a question blocks, they go on under the recommended option, and an answer given before then
// is the one they go on under. The loop never waits on a question, except one that changes a standing rule.
func byRecommendation(document Plan) (Plan, bool) {
	decided := slices.Clone(document.Decisions)
	taken := false
	for at := range decided {
		if decided[at].Answer == "" && !decided[at].answeredByHand() {
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
		if decision.Answer == "" && !decision.answeredByHand() && slices.Contains(decision.Blocks, item) {
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
			(decision.Answer != "" || decision.answeredByHand())
	})
	if at < 0 {
		return Decision{}, false
	}
	return document.Decisions[at], true
}

// parkedNote says which rows wait for a commit.
func parkedNote(document Plan) []string {
	var notes []string
	if len(document.parked) != 0 {
		notes = append(notes, "parked until a commit: "+strings.Join(document.parked, ", "))
	}
	return notes
}

// holds says why a holding decision holds its rows.
func (decision Decision) holds() string {
	switch {
	case decision.Answer == "" && decision.OwnersAct:
		return "its answer is an act only the owner takes, so it waits for the owner's answer while other rows run"
	case decision.Answer == "":
		// The lead answers it, never the owner (owner 2026-11-13, "don't wait for answers from me"): a session read "waits
		// for the owner's answer" here and sat 8 hours on capacity-record-history (doctrine-answer).
		return "it changes a standing rule, so no unattended run takes its recommendation: the lead answers it now (go run ./cmd/plan -answer -title <option> " + decision.ID + "), and its rows go on"
	}
	return "answered " + strconv.Quote(decision.Answer) + ", and the row was written for another answer; rewrite or retire it"
}
