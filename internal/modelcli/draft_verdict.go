package modelcli

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/inferencesurface"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
)

const (
	// draftVerdictMediaType names the record of served models' draft verdicts.
	draftVerdictMediaType = "application/vnd.overgo.draft-verdicts+json"
	// draftVerdictSchema versions its shape.
	draftVerdictSchema = "overgo/draft-verdicts/v1"
	// draftVerdictAlias is the store's pointer to the newest record.
	draftVerdictAlias = "inference/draft-verdicts/latest"
)

// draftVerdict is one judgement of a served model's own drafting (inference.Runner.JudgeOwnDrafts): whether drafting
// was refused for the model, under the build's execution identity of the runner, on the card.
type draftVerdict struct {
	Model     artifact.ID `json:"model"`
	Execution string      `json:"execution"`
	Device    string      `json:"device"`
	Refused   bool        `json:"refused"`
}

// draftVerdicts is the record: the verdicts the served opens reached, one a model, build and card.
type draftVerdicts struct {
	Rows []draftVerdict `json:"rows"`
	ID   artifact.ID    `json:"-"`
}

var draftVerdictCodec = artifact.JSONDocumentCodec(
	"draft verdicts", artifact.KindEvidence, draftVerdictMediaType, draftVerdictSchema,
	validateDraftVerdicts, func(value draftVerdicts) artifact.ID { return value.ID },
	func(value *draftVerdicts, id artifact.ID) { value.ID = id },
)

func validateDraftVerdicts(value *draftVerdicts) error {
	if value == nil {
		return errors.New("modelcli: nil draft verdicts")
	}
	for index, row := range value.Rows {
		if !row.Model.Valid() || row.Execution == "" || row.Device == "" {
			return errors.New("modelcli: incomplete draft verdict")
		}
		if index > 0 && compareDraftVerdicts(value.Rows[index-1], row) >= 0 {
			return errors.New("modelcli: unordered draft verdicts")
		}
	}
	return nil
}

func compareDraftVerdicts(left, right draftVerdict) int {
	return strings.Compare(left.Model.String()+"\x00"+left.Execution+"\x00"+left.Device, right.Model.String()+"\x00"+right.Execution+"\x00"+right.Device)
}

// recorded is the verdict the record holds for the subject's model, build and card, or nil.
func (record draftVerdicts) recorded(subject draftVerdict) *bool {
	at := slices.IndexFunc(record.Rows, func(row draftVerdict) bool { return compareDraftVerdicts(row, subject) == 0 })
	if at < 0 {
		return nil
	}
	return &record.Rows[at].Refused
}

// readDraftVerdicts is the store's record, empty where none is recorded or it cannot be read: a verdict missing is
// reached again by the judgement it would have spared.
func readDraftVerdicts(ctx context.Context, reader artifact.Reader) draftVerdicts {
	record, found, err := draftVerdictCodec.Resolve(ctx, reader, draftVerdictAlias)
	if err != nil || !found {
		return draftVerdicts{}
	}
	return record
}

// judgeServedDrafts judges the served model's own drafting by the verdict the store records for its model, build and
// card where one is recorded, so its open decodes once and probes nothing, and records the verdict a judgement reaches.
// A build that counted no execution identity (a plain go run) judges at every open, as before.
func judgeServedDrafts(ctx context.Context, repository string, runner *inference.Runner, model artifact.ID, record draftVerdicts) error {
	judged := processmeasure.Stage(ctx, "draft judgement")
	defer judged()
	execution, held := inferencesurface.Execution(reflect.TypeFor[inference.Runner]().PkgPath())
	device, err := runner.DeviceUUID(ctx)
	if err != nil {
		return err
	}
	subject := draftVerdict{Model: model, Execution: execution, Device: device}
	var verdict *bool
	if held {
		verdict = record.recorded(subject)
	}
	refused, reached, err := runner.JudgeOwnDrafts(ctx, verdict)
	if err != nil || !reached || !held {
		return err
	}
	subject.Refused = refused
	// A verdict the store cannot take now, another writer holding it among the causes, is reached again at the next open.
	_ = publishDraftVerdict(ctx, repository, subject)
	return nil
}

// publishDraftVerdict records verdict beside the others the store records when it writes, so a verdict another open
// recorded since this one read the record is kept, and moves the alias to the new record.
func publishDraftVerdict(ctx context.Context, repository string, verdict draftVerdict) error {
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	rows := slices.DeleteFunc(readDraftVerdicts(ctx, store).Rows, func(row draftVerdict) bool { return compareDraftVerdicts(row, verdict) == 0 })
	rows = append(rows, verdict)
	slices.SortFunc(rows, compareDraftVerdicts)
	next, err := draftVerdictCodec.New(draftVerdicts{Rows: rows})
	if err != nil {
		return err
	}
	var previous *artifact.ID
	if current, bound, err := artifact.ResolveAlias(ctx, store, draftVerdictAlias); err != nil {
		return err
	} else if bound {
		previous = &current
	}
	batch, err := draftVerdictCodec.Batch("inference/draft-verdicts/"+next.ID.String(), next, nil,
		[]artifact.AliasBinding{artifact.AliasMoveFrom(draftVerdictAlias, next.ID, previous)})
	if err != nil {
		return err
	}
	_, err = artifact.Publish(ctx, store, batch)
	return err
}
