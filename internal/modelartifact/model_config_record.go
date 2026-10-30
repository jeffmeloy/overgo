package modelartifact

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// architectureSource is the source file that declares a model's architecture,
// its shapes and its vocabulary, without which recorded sources do not say
// which model they are.
const architectureSource = "config.json"

// SourcesKey is recorded sources as one comparable text, file for file in name
// order, or empty where they do not declare the model's architecture
// (architectureSource): two models whose keys are equal and not empty were
// built from the same files. A sampling card alone names no model; Qwen3.5-4B
// and 9B ship the same one.
func SourcesKey(sources []ConfigSource) string {
	if !slices.ContainsFunc(sources, func(source ConfigSource) bool { return source.Name == architectureSource }) {
		return ""
	}
	files := make([]string, 0, len(sources))
	for _, source := range sources {
		files = append(files, source.Name+"="+source.SHA256)
	}
	slices.Sort(files)
	return strings.Join(files, "\n")
}

// ConfigSources is each model's recorded sources, from the newest model config
// the store holds for it: the files of the directory the model was converted
// or registered from, each with its digest.
func ConfigSources(ctx context.Context, store overgodb.DocumentReader) (map[artifact.ID][]ConfigSource, error) {
	sources := map[artifact.ID][]ConfigSource{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindProfile, MediaType: ModelConfigMediaType, Schema: ModelConfigSchema}},
		Order:     overgodb.DocumentNewestFirst,
	}, ParseModelConfigDocument, func(_ overgodb.DocumentView, document ModelConfigDocument) error {
		if _, seen := sources[document.Model]; !seen {
			sources[document.Model] = document.Sources
		}
		return nil
	})
	return sources, err
}

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
