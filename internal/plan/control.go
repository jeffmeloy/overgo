package plan

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/strictjson"
	"overgo/internal/worklease"

	"overgo/internal/artifact"
)

const (
	controlVersion   uint16 = 1
	controlMediaType        = "application/vnd.overgo.automation-control+json"
	controlSchema           = "overgo/automation-control/v1"
)

var containmentReasons = map[string]bool{
	"evidence-corruption": true, "evaluator-contamination": true,
	"worktree-collision": true, "device-instability": true,
	"rollback-unavailable": true,
}

// ControlEvent records immutable scoped operator decisions.
type ControlEvent struct {
	Version    uint16      `json:"version"`
	Kind       string      `json:"kind"`
	Lane       string      `json:"lane"`
	ReasonCode string      `json:"reason_code"`
	Detail     string      `json:"detail"`
	CodeCommit string      `json:"code_commit"`
	ID         artifact.ID `json:"-"`
	Worktree   string      `json:"worktree,omitzero"`
	Worker     string      `json:"worker,omitzero"`
	Mode       string      `json:"mode,omitzero"`
	Task       string      `json:"task,omitzero"`
	Previous   artifact.ID `json:"previous,omitzero"`
	Legacy     artifact.ID `json:"legacy,omitzero"`
}

var controlCodec = func() artifact.DocumentCodec[ControlEvent] {
	codec := artifact.JSONDocumentCodec("automation control", artifact.KindEvidence, controlMediaType, controlSchema,
		canonicalizeControl, func(value ControlEvent) artifact.ID { return value.ID },
		func(value *ControlEvent, id artifact.ID) { value.ID = id }, nil)
	codec.ContractFor = controlContract
	return codec
}()

// RecordControlEvent: validate and commit.
func RecordControlEvent(ctx context.Context, repository artifact.Repository, event ControlEvent) (ControlEvent, error) {
	event.Version = controlVersion
	if isStopControl(event.Kind) {
		return recordStopControl(ctx, repository, event)
	}
	identified, err := controlCodec.New(event)
	if err != nil {
		return ControlEvent{}, err
	}
	batch, err := controlCodec.Batch("automation/control/"+identified.ID.String(), identified, nil, nil)
	if err != nil {
		return ControlEvent{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return identified, err
}

// ReadControlEvent returns false when id is not automation-control evidence.
func ReadControlEvent(ctx context.Context, reader artifact.Reader, id artifact.ID) (ControlEvent, bool, error) {
	return worklease.ReadTypedDocument(ctx, reader, id, controlCodec.Contract, controlCodec.Read, controlContract(ControlEvent{Version: artifact.SecondDocumentVersion}))
}

func canonicalizeControl(event *ControlEvent) error {
	if event == nil || event.Version != controlVersion && event.Version != artifact.SecondDocumentVersion || !worklease.ValidAutomationText(event.Lane) ||
		!worklease.ValidAutomationText(event.ReasonCode) || !validAutomationDetail(event.Detail) ||
		!worklease.ValidCommit(event.CodeCommit) {
		return errors.New("plan: invalid automation control event")
	}
	if isStopControl(event.Kind) {
		return validateStopControl(*event)
	}
	if event.Version != controlVersion || event.Worktree != "" || event.Worker != "" || event.Mode != "" || event.Task != "" || event.Previous.Valid() || event.Legacy.Valid() {
		return errors.New("plan: legacy control contains stop fields")
	}
	switch event.Kind {
	case "override":
		if event.ReasonCode != "forced-advance" {
			return errors.New("plan: invalid override event")
		}
	case "containment":
		if !containmentReasons[event.ReasonCode] {
			return fmt.Errorf("plan: invalid containment reason %q", event.ReasonCode)
		}
	default:
		return errors.New("plan: invalid automation control kind")
	}
	return nil
}

func (event ControlEvent) Content() (artifact.Content, error) { return controlCodec.Content(event) }

const (
	stopControlSchema = "overgo/automation-control/v2"
	stopAliasRoot     = "automation/stop/"
	legacyStopPath    = "docs/plan_stop.json"
	// AutomationModeEnvironment follows the driver into dispatch and gate subprocesses.
	AutomationModeEnvironment = "OVERGO_AUTOMATION_MODE"
	// AutomationMaintenanceEnvironment names an exact retained maintenance grant.
	AutomationMaintenanceEnvironment = "OVERGO_AUTOMATION_MAINTENANCE"
	// ExecutionInteractive identifies supervised work; the loop sets unattended explicitly.
	ExecutionInteractive = "interactive"
	// ExecutionUnattended cannot consume a supervised maintenance grant.
	ExecutionUnattended = "unattended"
	// ExecutionAll scopes an operator stop to both execution modes.
	ExecutionAll = "all"
)

func isStopControl(kind string) bool {
	return slices.Contains([]string{ControlStop, ControlResume, ControlMaintenance}, kind)
}

func controlContract(event ControlEvent) artifact.DocumentContract {
	schema := controlSchema
	if event.Version == artifact.SecondDocumentVersion {
		schema = stopControlSchema
	}
	return artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: controlMediaType, Schema: schema}
}

func validateStopControl(event ControlEvent) error {
	if event.Version != artifact.SecondDocumentVersion || !filepath.IsAbs(event.Worktree) || strings.Contains(event.Worktree, worklease.NonCanonicalPathSeparator) || !worklease.ValidAutomationText(event.Worker) || event.Worker == worklease.UnassignedRole || !validStopMode(event.Mode) {
		return errors.New("plan: stop control requires exact worktree, owner, worker and execution scope")
	}
	if event.Previous.Valid() && event.Previous.Kind() != artifact.KindEvidence || event.Legacy.Valid() && event.Legacy.Kind() != artifact.KindEvidence {
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
	}
	return nil
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

// StopStatus projects durable authority; a changed HEAD never grants resumption.
// Invalid authority blocks execution while remaining readable for repair.
type StopStatus struct {
	State          string        `json:"state"`
	ID             artifact.ID   `json:"id,omitzero"`
	Event          *ControlEvent `json:"event,omitempty"`
	CurrentHead    string        `json:"current_head,omitzero"`
	LegacyMismatch bool          `json:"legacy_head_mismatch,omitzero"`
	Blocked        bool          `json:"blocked"`
	Problem        string        `json:"problem,omitzero"`
}

func stopAlias(root string) string {
	return worklease.Alias(stopAliasRoot, strings.ToLower(filepath.ToSlash(root)))
}

// ReadStop uses an existing reader when available. Standalone hooks may pass nil.
// Legacy bytes are read only; typed decisions refer to them without deleting them.
func ReadStop(ctx context.Context, reader artifact.Reader, root, head, mode string) (status StopStatus, err error) {
	mode, err = ExecutionMode(mode)
	if err != nil {
		return status, err
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return status, err
	}
	root = filepath.ToSlash(root)
	if reader == nil {
		store, openErr := overgodb.OpenReadOnly(filepath.Join(root, gitauthority.CanonicalOvergoDBDirectory))
		if openErr != nil {
			return status, openErr
		}
		defer func() { err = errors.Join(err, store.Close()) }()
		reader = store
	}
	// A long-lived reader must observe committed stop/resume events before deciding.
	if store, ok := reader.(*overgodb.Store); ok {
		if err := store.Refresh(ctx); err != nil {
			return status, err
		}
	}
	raw, readErr := os.ReadFile(filepath.Join(root, legacyStopPath))
	if readErr != nil && !os.IsNotExist(readErr) {
		return StopStatus{State: stopInvalid, Blocked: true, Problem: readErr.Error()}, nil
	}
	var legacy artifact.ID
	if readErr == nil {
		legacy, err = artifact.IdentifyBytes(artifact.KindEvidence, raw)
		if err != nil {
			return status, err
		}
	}
	id, found, err := reader.ResolveAlias(ctx, stopAlias(root))
	if err != nil {
		return status, err
	}
	status = StopStatus{State: stopAbsent, CurrentHead: head}
	if found {
		event, ok, readErr := ReadControlEvent(ctx, reader, id)
		if readErr != nil || !ok || event.Kind != ControlStop && event.Kind != ControlResume || !strings.EqualFold(event.Worktree, root) {
			return StopStatus{State: stopInvalid, ID: id, Blocked: true, Problem: "stop alias does not name valid scoped control"}, nil
		}
		status.ID = id
		status.Event = &event
		status.State = stopActive
		if event.Kind == ControlResume {
			status.State = stopResumed
		}
		if event.Legacy != legacy {
			status.State = stopInvalid
			status.Problem = "legacy marker changed after its recorded migration"
		}
	} else if legacy.Valid() {
		var marker struct {
			Reason string `json:"reason"`
			Head   string `json:"head"`
		}
		status.ID = legacy
		status.State = stopLegacyActive
		decodeErr := strictjson.DecodeBytes(raw, &marker)
		reason, detail, reasonErr := ParseStopReason(marker.Reason)
		if decodeErr != nil || reasonErr != nil || !worklease.ValidCommit(marker.Head) {
			status.State = stopInvalid
			status.Problem = "legacy stop marker is malformed; explicitly review its exact identity before recovery"
		}
		status.Event = &ControlEvent{Kind: ControlStop, ReasonCode: strings.TrimSpace(reason), Detail: strings.TrimSpace(detail), CodeCommit: marker.Head, Worktree: root, Mode: ExecutionAll, Legacy: legacy}
	}
	if status.Event != nil && head == "" {
		_, current, headErr := resolveCompletionRevision(ctx, root, gitHeadRevision)
		if headErr == nil {
			status.CurrentHead = current
		}
	}
	if status.State == stopLegacyActive {
		status.LegacyMismatch = status.Event.CodeCommit != status.CurrentHead
	}
	status.Blocked = status.State == stopInvalid || (status.State == stopActive || status.State == stopLegacyActive) && (status.Event.Mode == ExecutionAll || status.Event.Mode == mode)
	return status, nil
}

// String keeps stop provenance and explicit recovery visible in command output.
func (status StopStatus) String() string {
	if status.Event == nil {
		return fmt.Sprintf("stop=%s id=%s %s", status.State, status.ID, status.Problem)
	}
	legacy := ""
	if status.LegacyMismatch {
		legacy = "; legacy HEAD mismatch would previously lapse the stop; explicit resume remains required"
	}
	return fmt.Sprintf("stop=%s id=%s owner=%s worker=%s scope=%s worktree=%s bound_head=%s current_head=%s reason=%s: %s%s %s", status.State, status.ID, status.Event.Lane, status.Event.Worker, status.Event.Mode, status.Event.Worktree, status.Event.CodeCommit, status.CurrentHead, status.Event.ReasonCode, status.Event.Detail, legacy, status.Problem)
}

// RequireExecution checks a current stop and an optional exact maintenance grant.
// Maintenance does not move the stop alias and cannot authorize unattended work.
func (status StopStatus) RequireExecution(ctx context.Context, reader artifact.Reader, role, worker, mode, task, grant string) error {
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
	event, found, err := ReadControlEvent(ctx, reader, id)
	if err != nil {
		return err
	}
	if !found || event.Kind != ControlMaintenance || status.Event == nil || status.State != stopActive && status.State != stopLegacyActive || event.Previous != status.ID || !strings.EqualFold(event.Worktree, status.Event.Worktree) || event.Lane != role || event.Worker != worker || event.Task != task || event.CodeCommit != status.CurrentHead || mode != ExecutionInteractive {
		return errors.New("plan: maintenance requires its exact active stop, supervised worker, task and HEAD")
	}
	return nil
}

func recordStopControl(ctx context.Context, repository artifact.Repository, event ControlEvent) (ControlEvent, error) {
	root, err := filepath.Abs(event.Worktree)
	if err != nil {
		return ControlEvent{}, err
	}
	event.Worktree = filepath.ToSlash(root)
	current, err := ReadStop(ctx, repository, root, event.CodeCommit, ExecutionInteractive)
	if err != nil {
		return ControlEvent{}, err
	}
	if event.Previous != current.ID {
		return ControlEvent{}, errors.New("plan: stop changed; review its current identity before retrying")
	}
	if current.Event != nil && current.Event.Version >= artifact.InitialDocumentVersion && current.State != stopResumed && (event.Lane != current.Event.Lane || event.Mode != current.Event.Mode && !(event.Kind == ControlStop && event.Mode == ExecutionAll)) {
		return ControlEvent{}, errors.New("plan: stop owner or scope mismatch")
	}
	if event.Kind != ControlStop && current.Event != nil && event.Mode != current.Event.Mode {
		return ControlEvent{}, errors.New("plan: explicit resume preserves the exact stop scope")
	}
	if event.Kind != ControlStop && (current.Event == nil || current.State == stopResumed || current.State == stopInvalid) {
		return ControlEvent{}, errors.New("plan: resume or maintenance requires a valid active stop")
	}

	if current.Event != nil {
		event.Legacy = current.Event.Legacy
	}
	event.Version = artifact.SecondDocumentVersion
	identified, err := controlCodec.New(event)
	if err != nil {
		return ControlEvent{}, err
	}
	var previous *artifact.ID
	var lineage []artifact.Lineage
	if current.ID.Valid() {
		lineage = append(lineage, artifact.Lineage{Child: identified.ID, Parent: current.ID, Relation: artifact.RelationDerivedFrom})
	}
	if current.Event != nil && current.Event.Version >= artifact.InitialDocumentVersion {
		previous = &current.ID
	}
	var aliases []artifact.AliasBinding
	if event.Kind != ControlMaintenance {
		aliases = []artifact.AliasBinding{{Name: stopAlias(root), Target: identified.ID, Previous: previous}}
	}
	batch, err := controlCodec.Batch("automation/control/"+identified.ID.String(), identified, lineage, aliases)
	if err != nil {
		return ControlEvent{}, err
	}
	if current.Event != nil && current.Event.Version < artifact.InitialDocumentVersion {
		raw, readErr := os.ReadFile(filepath.Join(root, legacyStopPath))
		if readErr != nil {
			return ControlEvent{}, readErr
		}
		id, identifyErr := artifact.IdentifyBytes(artifact.KindEvidence, raw)
		if identifyErr != nil {
			return ControlEvent{}, identifyErr
		}
		if id != current.ID {
			return ControlEvent{}, errors.New("plan: legacy stop changed during migration")
		}
		content, contentErr := (artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: "application/json", Schema: "overgo/legacy-plan-stop/v1"}).Content(id, raw)
		if contentErr != nil {
			return ControlEvent{}, contentErr
		}
		batch.Contents = append(batch.Contents, content)
	}
	if _, err = artifact.CommitBatch(ctx, repository, batch); err != nil {
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
	stopAbsent       = "absent"
	stopInvalid      = "invalid"
	stopActive       = "active"
	stopLegacyActive = "legacy-active"
	stopResumed      = "resumed"
)

// Legacy CLI stop reasons encode category and detail with a colon.
const stopReasonSeparator = ":"
