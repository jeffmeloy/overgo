package runrecord

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	// StageReceiptMediaType identifies stage receipt documents.
	StageReceiptMediaType = "application/vnd.overgo.stage-receipt+json"
	// StageReceiptSchema identifies the stage receipt contract.
	StageReceiptSchema = "overgo/stage-receipt/v1"
	// StageReceiptAliasRoot scopes current workflow stage receipts.
	StageReceiptAliasRoot = "stage-receipt/"
)

// StageState defines durable node lifecycle.
type StageState string

const (
	// StageAdmitted marks accepted stage work.
	StageAdmitted StageState = "admitted"
	// StageRunning marks executing stage work.
	StageRunning StageState = "running"
	// StageWaiting marks recoverable blocked work.
	StageWaiting StageState = "waiting"
	// StageCompleted marks durable stage output.
	StageCompleted StageState = "completed"
	// StageFailed marks terminal failed work.
	StageFailed StageState = "failed"
)

// StageBinding records ordered artifacts at one typed port.
type StageBinding struct {
	Port      recipe.PortName `json:"port"`
	Artifacts []artifact.ID   `json:"artifacts,omitempty"`
}

// StageReceipt records one immutable node transition.
type StageReceipt struct {
	Version   uint16         `json:"version"`
	Recipe    artifact.ID    `json:"recipe"`
	Node      recipe.NodeID  `json:"node"`
	Operation artifact.ID    `json:"operation"`
	Attempt   uint32         `json:"attempt"`
	State     StageState     `json:"state"`
	Inputs    []StageBinding `json:"inputs,omitempty"`
	Outputs   []StageBinding `json:"outputs,omitempty"`
	Previous  artifact.ID    `json:"previous,omitzero"`
	Failure   string         `json:"failure,omitempty"`
	ID        artifact.ID    `json:"-"`
}

var stageReceiptCodec = artifact.JSONDocumentCodec(
	"stage receipt", artifact.KindEvidence, StageReceiptMediaType, StageReceiptSchema,
	canonicalizeStageReceipt,
	func(value StageReceipt) artifact.ID { return value.ID },
	func(value *StageReceipt, id artifact.ID) { value.ID = id },
	cloneStageReceipt,
)

// NewStageReceipt identifies one validated node transition.
func NewStageReceipt(value StageReceipt) (StageReceipt, error) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return stageReceiptCodec.New(value)
}

// ParseStageReceipt decodes and validates one immutable stage transition.
func ParseStageReceipt(content []byte) (StageReceipt, error) {
	return stageReceiptCodec.Parse(content)
}

// ResolveStageReceipt returns the latest receipt for one operation node.
func ResolveStageReceipt(
	ctx context.Context,
	reader artifact.Reader,
	operation artifact.ID,
	node recipe.NodeID,
) (StageReceipt, bool, error) {
	id, found, err := artifact.ResolveAlias(ctx, reader, stageReceiptAlias(operation, node))
	if err != nil || !found {
		return StageReceipt{}, found, err
	}
	value, err := stageReceiptCodec.Require(ctx, reader, id)
	return value, err == nil, err
}

// PublishStageReceipt advances one operation-node lifecycle.
func PublishStageReceipt(
	ctx context.Context,
	repository artifact.Repository,
	value StageReceipt,
	contents []artifact.Content,
	descriptors []artifact.Descriptor,
) (StageReceipt, error) {
	if ctx == nil || repository == nil {
		return StageReceipt{}, errors.New("run record: stage receipt repository is absent")
	}
	previous, found, err := ResolveStageReceipt(ctx, repository, value.Operation, value.Node)
	if err != nil {
		return StageReceipt{}, err
	}
	if found {
		value.Previous = previous.ID
		if !stageTransition(previous, value) {
			return StageReceipt{}, errors.New("run record: invalid stage transition")
		}
	}
	value, err = NewStageReceipt(value)
	if err != nil {
		return StageReceipt{}, err
	}
	receiptContent, err := stageReceiptCodec.Content(value)
	if err != nil {
		return StageReceipt{}, err
	}
	contents = append(slices.Clone(contents), receiptContent)
	parents := []artifact.ID{value.Recipe, value.Operation, value.Previous}
	for _, bindings := range [][]StageBinding{value.Inputs, value.Outputs} {
		for _, binding := range bindings {
			parents = append(parents, binding.Artifacts...)
		}
	}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	alias := artifact.AliasBinding{Name: stageReceiptAlias(value.Operation, value.Node), Target: value.ID}
	if found {
		alias.Previous = artifact.IDPointer(previous.ID)
	}
	batch, err := artifact.NewDocumentBatch(
		"stage-receipt/"+value.ID.String(), contents,
		artifact.DependencyLineage(value.ID, parents...),
		[]artifact.AliasBinding{alias},
	)
	if err != nil {
		return StageReceipt{}, err
	}
	batch.Artifacts = append(batch.Artifacts, descriptors...)
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return StageReceipt{}, err
	}
	return value, nil
}

func canonicalizeStageReceipt(value *StageReceipt) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Node == "" ||
		value.Operation.Kind() != artifact.KindEvidence || value.Attempt == 0 || !value.State.valid() ||
		(value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence) {
		return errors.New("run record: invalid stage receipt")
	}
	if value.State == StageFailed && value.Failure == "" || value.State != StageFailed && value.Failure != "" {
		return errors.New("run record: invalid stage failure")
	}
	for _, bindings := range [][]StageBinding{value.Inputs, value.Outputs} {
		for _, binding := range bindings {
			if binding.Port == "" || slices.ContainsFunc(binding.Artifacts, func(id artifact.ID) bool { return !id.Valid() }) {
				return errors.New("run record: invalid stage binding")
			}
		}
	}
	return nil
}

func (state StageState) valid() bool {
	return state == StageAdmitted || state == StageRunning || state == StageWaiting ||
		state == StageCompleted || state == StageFailed
}

func stageTransition(previous, next StageReceipt) bool {
	if previous.Recipe != next.Recipe || previous.Node != next.Node || previous.Operation != next.Operation {
		return false
	}
	if next.Attempt == previous.Attempt {
		return previous.State == StageAdmitted && next.State == StageRunning ||
			previous.State == StageRunning && (next.State == StageWaiting || next.State == StageCompleted || next.State == StageFailed)
	}
	// A new attempt may open from admitted too: a crash between the
	// persisted admission and the persisted running state must leave a
	// recoverable stage, not one wedged before its first heartbeat.
	return next.Attempt == previous.Attempt+1 &&
		(previous.State == StageAdmitted || previous.State == StageRunning ||
			previous.State == StageWaiting || previous.State == StageFailed) &&
		next.State == StageAdmitted
}

func cloneStageReceipt(value StageReceipt) StageReceipt {
	value.Inputs = cloneStageBindings(value.Inputs)
	value.Outputs = cloneStageBindings(value.Outputs)
	return value
}

func cloneStageBindings(values []StageBinding) []StageBinding {
	values = slices.Clone(values)
	for index := range values {
		values[index].Artifacts = slices.Clone(values[index].Artifacts)
	}
	return values
}

func stageReceiptAlias(operation artifact.ID, node recipe.NodeID) string {
	return StageReceiptAliasRoot + operation.String() + "/" + string(node)
}
