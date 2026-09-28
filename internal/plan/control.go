package plan

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/atomicfile"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
	"overgo/internal/strictjson"
	"overgo/internal/worklease"
)

// stopFile holds a worktree's current stop or resume and the maintenance
// grants under it, in the checkout's process state. It is operational state,
// kept out of the evidence store: a stop neither churns the store every
// store-reading suite depends on, nor needs the store opened to be read.
const stopFile = "stop.json"

// stopLock serializes stop writers, so a writer that has not seen the
// current decision cannot replace it.
const stopLock = "stop.lock"

// stopFileMode keeps the stop state private to the operator.
const stopFileMode = 0o600

const (
	// AutomationModeEnvironment follows the driver into dispatch and gate subprocesses.
	AutomationModeEnvironment = "OVERGO_AUTOMATION_MODE"
	// AutomationMaintenanceEnvironment names an exact maintenance grant.
	AutomationMaintenanceEnvironment = "OVERGO_AUTOMATION_MAINTENANCE"
	// ExecutionInteractive identifies supervised work; the loop sets unattended explicitly.
	ExecutionInteractive = "interactive"
	// ExecutionUnattended cannot consume a supervised maintenance grant.
	ExecutionUnattended = "unattended"
	// ExecutionAll scopes an operator stop to both execution modes.
	ExecutionAll = "all"
)

// ControlEvent is one operator decision on a worktree's execution: a stop,
// the resume that retires it, or a maintenance grant for one task under it.
type ControlEvent struct {
	Kind       string      `json:"kind"`
	Lane       string      `json:"lane"`
	ReasonCode string      `json:"reason_code"`
	Detail     string      `json:"detail"`
	CodeCommit string      `json:"code_commit"`
	ID         artifact.ID `json:"-"`
	Worktree   string      `json:"worktree"`
	Worker     string      `json:"worker"`
	Mode       string      `json:"mode"`
	Task       string      `json:"task,omitzero"`
	Previous   artifact.ID `json:"previous,omitzero"`
}

// stopState is the stop file: the current stop or resume, and the
// maintenance grants issued under that stop.
type stopState struct {
	Current     ControlEvent   `json:"current"`
	Maintenance []ControlEvent `json:"maintenance,omitempty"`
}

func validateControl(event ControlEvent) error {
	if !worklease.ValidAutomationText(event.Lane) || !worklease.ValidAutomationText(event.ReasonCode) || !validAutomationDetail(event.Detail) ||
		!worklease.ValidCommit(event.CodeCommit) || !filepath.IsAbs(event.Worktree) || strings.Contains(event.Worktree, worklease.NonCanonicalPathSeparator) ||
		!worklease.ValidAutomationText(event.Worker) || event.Worker == worklease.UnassignedRole || !validStopMode(event.Mode) {
		return errors.New("plan: stop control requires exact worktree, owner, worker and execution scope")
	}
	if event.Previous.Valid() && event.Previous.Kind() != artifact.KindEvidence {
		return errors.New("plan: invalid stop predecessor")
	}
	switch event.Kind {
	case ControlStop:
		if event.Task != "" {
			return errors.New("plan: stop does not name a maintenance task")
		}
		_, _, err := ParseStopReason(event.ReasonCode + ": " + event.Detail)
		return err
	case ControlResume:
		if event.ReasonCode != "operator-resume" || !event.Previous.Valid() || event.Task != "" {
			return errors.New("plan: resume requires an exact operator stop")
		}
	case ControlMaintenance:
		if event.ReasonCode != "operator-maintenance" || !event.Previous.Valid() || !strings.Contains(event.Task, stepReferenceSeparator) || !worklease.ValidAutomationText(event.Task) {
			return errors.New("plan: maintenance requires an exact stop and task")
		}
	default:
		return errors.New("plan: invalid stop control kind")
	}
	return nil
}

// identify gives an event its content identity, the ID an operator names to
// resume a stop or to use a maintenance grant.
func identify(event ControlEvent) (ControlEvent, error) {
	id, err := artifact.JSONID(artifact.KindEvidence, event)
	event.ID = id
	return event, err
}

func validStopMode(mode string) bool {
	return slices.Contains([]string{ExecutionAll, ExecutionInteractive, ExecutionUnattended}, mode)
}

// ExecutionMode resolves the driver mode without inferring supervision from a PID.
func ExecutionMode(explicit string) (string, error) {
	mode := cmp.Or(strings.TrimSpace(explicit), strings.TrimSpace(os.Getenv(AutomationModeEnvironment)), ExecutionInteractive)
	if mode != ExecutionInteractive && mode != ExecutionUnattended {
		return "", errors.New("plan: execution mode must be interactive or unattended")
	}
	return mode, nil
}

// StopStatus projects the worktree's stop; a changed HEAD never grants
// resumption. An unreadable stop file blocks execution while remaining
// readable for repair.
type StopStatus struct {
	State       string        `json:"state"`
	ID          artifact.ID   `json:"id,omitzero"`
	Event       *ControlEvent `json:"event,omitempty"`
	CurrentHead string        `json:"current_head,omitzero"`
	Blocked     bool          `json:"blocked"`
	Problem     string        `json:"problem,omitzero"`
	grants      []ControlEvent
}

// ReadStop reads the worktree's stop file.
func ReadStop(ctx context.Context, root, head, mode string) (StopStatus, error) {
	mode, err := ExecutionMode(mode)
	if err != nil {
		return StopStatus{}, err
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return StopStatus{}, err
	}
	root = filepath.ToSlash(root)
	raw, err := os.ReadFile(filepath.Join(root, processlock.StateDirectory, stopFile))
	if errors.Is(err, os.ErrNotExist) {
		return StopStatus{State: stopAbsent, CurrentHead: head}, nil
	}
	if err != nil {
		return StopStatus{State: stopInvalid, Blocked: true, Problem: err.Error()}, nil
	}
	var state stopState
	if err := strictjson.DecodeBytes(raw, &state); err != nil {
		return StopStatus{State: stopInvalid, Blocked: true, Problem: "stop file is malformed: " + err.Error()}, nil
	}
	current, err := identify(state.Current)
	if err != nil || validateControl(current) != nil || current.Kind == ControlMaintenance || !strings.EqualFold(current.Worktree, root) {
		return StopStatus{State: stopInvalid, Blocked: true, Problem: "stop file does not name a valid stop or resume for this worktree"}, nil
	}
	status := StopStatus{State: stopActive, ID: current.ID, Event: &current, CurrentHead: head}
	if current.Kind == ControlResume {
		status.State = stopResumed
	}
	for _, grant := range state.Maintenance {
		identified, err := identify(grant)
		if err != nil || validateControl(identified) != nil || identified.Kind != ControlMaintenance {
			return StopStatus{State: stopInvalid, ID: current.ID, Blocked: true, Problem: "stop file holds an invalid maintenance grant"}, nil
		}
		status.grants = append(status.grants, identified)
	}
	if head == "" {
		if _, resolved, headErr := resolveRevision(ctx, root, gitHeadRevision); headErr == nil {
			status.CurrentHead = resolved
		}
	}
	status.Blocked = status.State == stopActive && (current.Mode == ExecutionAll || current.Mode == mode)
	return status, nil
}

// String keeps stop provenance and explicit recovery visible in command output.
func (status StopStatus) String() string {
	if status.Event == nil {
		return fmt.Sprintf("stop=%s id=%s %s", status.State, status.ID, status.Problem)
	}
	return fmt.Sprintf("stop=%s id=%s owner=%s worker=%s scope=%s worktree=%s bound_head=%s current_head=%s reason=%s: %s %s", status.State, status.ID, status.Event.Lane, status.Event.Worker, status.Event.Mode, status.Event.Worktree, status.Event.CodeCommit, status.CurrentHead, status.Event.ReasonCode, status.Event.Detail, status.Problem)
}

// RequireExecution checks a current stop and an optional exact maintenance
// grant. Maintenance does not retire the stop and cannot authorize
// unattended work.
func (status StopStatus) RequireExecution(role, worker, mode, task, grant string) error {
	if status.State == stopInvalid {
		return errors.New(status.String())
	}
	if grant == "" {
		if status.Blocked {
			return errors.New(status.String())
		}
		return nil
	}
	id, err := artifact.ParseID(grant)
	if err != nil {
		return err
	}
	at := slices.IndexFunc(status.grants, func(event ControlEvent) bool { return event.ID == id })
	if at < 0 || status.State != stopActive {
		return errors.New("plan: maintenance requires its exact active stop, supervised worker, task and HEAD")
	}
	event := status.grants[at]
	if event.Previous != status.ID || event.Lane != role || event.Worker != worker || event.Task != task || event.CodeCommit != status.CurrentHead || mode != ExecutionInteractive {
		return errors.New("plan: maintenance requires its exact active stop, supervised worker, task and HEAD")
	}
	return nil
}

// RecordStop writes one operator decision to the worktree's stop file: a
// stop or resume replaces the current decision, a maintenance grant joins
// the stop it names. Each must name the decision it follows, so a stale
// operator decision is refused rather than overwriting a newer one.
func RecordStop(ctx context.Context, event ControlEvent) (ControlEvent, error) {
	root, err := filepath.Abs(event.Worktree)
	if err != nil {
		return ControlEvent{}, err
	}
	event.Worktree = filepath.ToSlash(root)
	if err := validateControl(event); err != nil {
		return ControlEvent{}, err
	}
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return ControlEvent{}, err
	}
	lock, err := processlock.AcquireContext(ctx, filepath.Join(root, processlock.StateDirectory, stopLock), stopFileMode)
	if err != nil {
		return ControlEvent{}, err
	}
	defer lock.Close()
	current, err := ReadStop(ctx, root, event.CodeCommit, ExecutionInteractive)
	if err != nil {
		return ControlEvent{}, err
	}
	if event.Previous != current.ID {
		return ControlEvent{}, errors.New("plan: stop changed; review its current identity before retrying")
	}
	if current.Event != nil && current.State != stopResumed && (event.Lane != current.Event.Lane || event.Mode != current.Event.Mode && !(event.Kind == ControlStop && event.Mode == ExecutionAll)) {
		return ControlEvent{}, errors.New("plan: stop owner or scope mismatch")
	}
	if event.Kind != ControlStop && (current.Event == nil || current.State != stopActive || event.Mode != current.Event.Mode) {
		return ControlEvent{}, errors.New("plan: resume or maintenance requires the exact scope of a valid active stop")
	}
	identified, err := identify(event)
	if err != nil {
		return ControlEvent{}, err
	}
	state := stopState{Current: identified}
	if identified.Kind == ControlMaintenance {
		state = stopState{Current: *current.Event, Maintenance: append(slices.Clone(current.grants), identified)}
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return ControlEvent{}, err
	}
	if err := atomicfile.Write(filepath.Join(root, processlock.StateDirectory, stopFile), append(data, '\n'), stopFileMode); err != nil {
		return ControlEvent{}, err
	}
	if identified.Kind == ControlStop {
		if err := processcontrol.NotifyCampaignStop(ctx, root); err != nil {
			return identified, fmt.Errorf("stop recorded, but live supervisor notification failed: %w", err)
		}
	}
	return identified, nil
}

var validStopReasons = []string{"user-stop", "irreversible", "external-prereq"}

// ParseStopReason validates and splits the declared stop vocabulary.
func ParseStopReason(reason string) (string, string, error) {
	reason = strings.TrimSpace(reason)
	matched := ""
	for _, v := range validStopReasons {
		if reason == v || strings.HasPrefix(reason, v+stopReasonSeparator) {
			matched = v
			break
		}
	}
	if matched == "" {
		return "", "", fmt.Errorf("invalid stop reason %q -- must begin with one of: user-stop: / irreversible: / external-prereq: <detail>", reason)
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
				return "", "", fmt.Errorf("stop refused: %q is self-pacing, not an external prerequisite -- continue the plan", selfPacing)
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
			return "", "", errors.New("stop refused: external-prereq detail names nothing external (no background task, owner action, merge, or running work) -- continue the plan")
		}
	}
	code, detail, _ := strings.Cut(reason, stopReasonSeparator)
	return strings.TrimSpace(code), strings.TrimSpace(detail), nil
}

// ControlStop records an operator stop; it never expires with HEAD.
const ControlStop = "stop"

// ControlResume explicitly retires one scoped stop.
const ControlResume = "resume"

// ControlMaintenance authorizes one supervised task without resuming the loop.
const ControlMaintenance = "maintenance"

const (
	stopAbsent  = "absent"
	stopInvalid = "invalid"
	stopActive  = "active"
	stopResumed = "resumed"
)

// CLI stop reasons encode category and detail with a colon.
const stopReasonSeparator = ":"
