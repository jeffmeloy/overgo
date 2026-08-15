package plan

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
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

// ControlEvent is immutable evidence of an explicit automation override or a
// lane-scoped containment action. It never grants promotion authority.
type ControlEvent struct {
	Version    uint16      `json:"version"`
	Kind       string      `json:"kind"`
	Lane       string      `json:"lane"`
	ReasonCode string      `json:"reason_code"`
	Detail     string      `json:"detail"`
	CodeCommit string      `json:"code_commit"`
	ID         artifact.ID `json:"-"`
}

var controlCodec = artifact.JSONDocumentCodec("automation control", artifact.KindEvidence, controlMediaType, controlSchema,
	canonicalizeControl, func(value ControlEvent) artifact.ID { return value.ID },
	func(value *ControlEvent, id artifact.ID) { value.ID = id }, nil)

// RecordControlEvent validates and commits one control event.
func RecordControlEvent(ctx context.Context, repository artifact.Repository, event ControlEvent) (ControlEvent, error) {
	event.Version = controlVersion
	identified, err := controlCodec.New(event)
	if err != nil {
		return ControlEvent{}, err
	}
	content, err := identified.Content()
	if err != nil {
		return ControlEvent{}, err
	}
	batch, err := artifact.NewDocumentBatch("automation/control/"+identified.ID.String(), []artifact.Content{content}, nil, nil)
	if err != nil {
		return ControlEvent{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return identified, err
}

// ReadControlEvent returns false when id is not automation-control evidence.
func ReadControlEvent(ctx context.Context, reader artifact.Reader, id artifact.ID) (ControlEvent, bool, error) {
	return readTypedDocument(ctx, reader, id, controlCodec.Contract, controlCodec.Read)
}

func canonicalizeControl(event *ControlEvent) error {
	if event == nil || event.Version != controlVersion || !textcheck.Bounded(event.Lane, 2048, "\x00\r\n") ||
		!textcheck.Bounded(event.ReasonCode, 2048, "\x00\r\n") || !textcheck.Bounded(event.Detail, 4096, "\x00\r\n") ||
		!validCommit(event.CodeCommit) {
		return errors.New("plan: invalid automation control event")
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
