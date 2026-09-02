package modelartifact

import (
	"context"
	"errors"
	"fmt"
	"os"

	"overgo/internal/artifact"
)

// IdentifyConvertedModel is the converted file's own identity, the one a
// fresh conversion records under before the store catalogs it.
func IdentifyConvertedModel(output string) (artifact.ID, error) {
	converted, err := os.Open(output)
	if err != nil {
		return artifact.ID{}, err
	}
	defer converted.Close()
	model, _, err := artifact.Identify(artifact.KindModel, converted)
	return model, err
}

// RecordModelConfig commits a checkpoint directory's inference- and
// training-relevant declarations as a typed store artifact bound to the
// given model identity. A directory declaring nothing commits nothing;
// that fact is reported, not padded. The one owner of the record: the
// converter records at conversion under the converted file's identity,
// and the model-config command records for a servable model under the
// identity the store already holds for its location.
func RecordModelConfig(ctx context.Context, repository artifact.Repository, source string, model artifact.ID) (artifact.ID, bool, error) {
	if ctx == nil || repository == nil || model.Kind() != artifact.KindModel {
		return artifact.ID{}, false, errors.New("model config: incomplete record request")
	}
	sequence, generation, sources, err := ReadModelConfigComponents(source)
	if err != nil {
		return artifact.ID{}, false, err
	}
	if sequence == nil && generation == nil {
		return artifact.ID{}, false, nil
	}
	document, err := NewModelConfigDocument(model, sequence, generation, sources)
	if err != nil {
		return artifact.ID{}, false, err
	}
	batch, err := document.Batch("model-config/" + document.ID.String())
	if err != nil {
		return artifact.ID{}, false, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return artifact.ID{}, false, fmt.Errorf("model config: commit: %w", err)
	}
	return document.ID, true, nil
}
