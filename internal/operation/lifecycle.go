package operation

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	lifecycleMediaType = "application/vnd.overgo.operation-lifecycle+json"
	lifecycleSchema    = "overgo/operation-lifecycle/v1"
	lifecycleAliasRoot = "operations/lifecycle/"
)

var ErrLifecycleConflict = errors.New("operation: lifecycle identity conflict")

type lifecycleRecord struct {
	Version   uint16        `json:"version"`
	Operation artifact.ID   `json:"operation"`
	Intent    artifact.ID   `json:"intent"`
	Task      recipe.Task   `json:"task"`
	Recipe    artifact.ID   `json:"recipe"`
	Run       artifact.ID   `json:"run"`
	Outputs   []artifact.ID `json:"outputs,omitempty"`
	ID        artifact.ID   `json:"-"`
}

var lifecycleCodec = artifact.JSONDocumentCodec(
	"operation lifecycle", artifact.KindEvidence, lifecycleMediaType, lifecycleSchema,
	canonicalizeLifecycle,
	func(value lifecycleRecord) artifact.ID { return value.ID },
	func(value *lifecycleRecord, id artifact.ID) { value.ID = id },
	func(value lifecycleRecord) lifecycleRecord {
		value.Outputs = append([]artifact.ID(nil), value.Outputs...)
		return value
	},
)

// ExecuteReentrant returns a committed completion when the same operation and
// intent are retried. A new execution is published only after every completion
// artifact exists, making those artifacts the durable postcondition.
func ExecuteReentrant(
	ctx context.Context,
	repository artifact.Repository,
	reporter Reporter,
	request Request,
	intent artifact.ID,
	execute func(context.Context) (Completion, error),
) (Completion, error) {
	if ctx == nil || repository == nil || reporter == nil || execute == nil ||
		!request.Task.Valid() || request.Recipe.Kind() != artifact.KindRecipe || !intent.Valid() {
		return Completion{}, errors.New("operation: invalid reentrant lifecycle")
	}
	operationID := reporter.OperationID()
	if !operationID.Valid() {
		return execute(ctx)
	}
	if completion, found, err := loadLifecycle(ctx, repository, operationID, request, intent); err != nil || found {
		return completion, err
	}
	completion, err := execute(ctx)
	if err != nil {
		return completion, err
	}
	if err := validateCompletionPostcondition(ctx, repository, completion); err != nil {
		return completion, err
	}
	record, err := lifecycleCodec.New(lifecycleRecord{
		Version: artifact.InitialDocumentVersion, Operation: operationID, Intent: intent,
		Task: request.Task, Recipe: request.Recipe, Run: completion.Run, Outputs: completion.Outputs,
	})
	if err != nil {
		return completion, err
	}
	return publishLifecycle(context.WithoutCancel(ctx), repository, record)
}

func loadLifecycle(
	ctx context.Context,
	repository artifact.Repository,
	operationID artifact.ID,
	request Request,
	intent artifact.ID,
) (Completion, bool, error) {
	record, found, err := lifecycleCodec.Resolve(ctx, repository, lifecycleAlias(operationID))
	if err != nil || !found {
		if err == nil {
			err = errors.New("operation: lifecycle record is absent")
		}
		return Completion{}, true, err
	}
	if record.Operation != operationID || record.Intent != intent || record.Task != request.Task || record.Recipe != request.Recipe {
		return Completion{}, true, ErrLifecycleConflict
	}
	completion := Completion{Run: record.Run, Outputs: append([]artifact.ID(nil), record.Outputs...)}
	if err := validateCompletionPostcondition(ctx, repository, completion); err != nil {
		return completion, true, err
	}
	return completion, true, nil
}

func publishLifecycle(ctx context.Context, repository artifact.Repository, record lifecycleRecord) (Completion, error) {
	completion := Completion{Run: record.Run, Outputs: append([]artifact.ID(nil), record.Outputs...)}
	for {
		if loaded, found, err := loadLifecycle(ctx, repository, record.Operation, Request{Task: record.Task, Recipe: record.Recipe}, record.Intent); err != nil || found {
			return loaded, err
		}
		head, _ := repository.Head()
		batch, err := lifecycleCodec.Batch(
			"operation/lifecycle/"+record.ID.String(), record,
			artifact.DependencyLineage(record.ID, append([]artifact.ID{record.Run}, record.Outputs...)...),
			[]artifact.AliasBinding{{Name: lifecycleAlias(record.Operation), Target: record.ID}},
		)
		if err != nil {
			return completion, err
		}
		batch.ExpectedHead = &head
		if _, err = artifact.CommitBatch(ctx, repository, batch); err == nil {
			return completion, nil
		}
		if !errors.Is(err, artifact.ErrCommitPrecondition) {
			return completion, err
		}
		if err := ctx.Err(); err != nil {
			return completion, err
		}
	}
}

func validateCompletionPostcondition(ctx context.Context, repository artifact.Reader, completion Completion) error {
	if err := validateCompletion(completion); err != nil {
		return err
	}
	for _, id := range append([]artifact.ID{completion.Run}, completion.Outputs...) {
		if _, found, err := repository.Artifact(ctx, id); err != nil {
			return err
		} else if !found {
			return fmt.Errorf("operation: lifecycle postcondition %s is absent", id)
		}
	}
	return nil
}

func canonicalizeLifecycle(value *lifecycleRecord) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || !value.Operation.Valid() ||
		!value.Intent.Valid() || !value.Task.Valid() || value.Recipe.Kind() != artifact.KindRecipe {
		return errors.New("operation: invalid lifecycle record")
	}
	completion := Completion{Run: value.Run, Outputs: value.Outputs}
	if err := validateCompletion(completion); err != nil {
		return err
	}
	value.Outputs = append([]artifact.ID(nil), value.Outputs...)
	return nil
}

func lifecycleAlias(operationID artifact.ID) string {
	return lifecycleAliasRoot + operationID.String()
}
