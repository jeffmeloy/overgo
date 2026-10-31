package modelartifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// architectureSource is the source file that declares a model's architecture,
// its shapes and its vocabulary.
const architectureSource = "config.json"

// SourcesKey is the digest of the architecture file among recorded sources
// (architectureSource), or empty where they hold none: two models whose keys
// are equal and not empty were built from the same checkpoint. The other
// files a record reads are not compared, since records of one model read
// different ones (Qwen3.5-4B's conversion recorded its config.json, and later
// a sampling card alone), and a sampling card names no model: Qwen3.5-4B and
// 9B ship the same one.
func SourcesKey(sources []ConfigSource) string {
	at := slices.IndexFunc(sources, func(source ConfigSource) bool { return source.Name == architectureSource })
	if at < 0 {
		return ""
	}
	return sources[at].SHA256
}

// weightSourcePrefix opens the name a recorded source weight file is kept under among a model's sources.
const weightSourcePrefix = "weights:"

// WeightsKey names the source weights a model's sources record, a digest of each weight file's name and digest; empty
// where none is recorded. A trained checkpoint keeps its base's config files, so the config alone (SourcesKey) would
// take one for the other; the weights tell them apart.
func WeightsKey(sources []ConfigSource) string {
	var lines []string
	for _, source := range sources {
		if strings.HasPrefix(source.Name, weightSourcePrefix) {
			lines = append(lines, source.Name+" "+source.SHA256)
		}
	}
	if len(lines) == 0 {
		return ""
	}
	slices.Sort(lines)
	digest := sha256.Sum256([]byte(strings.Join(lines, "\n")))
	return hex.EncodeToString(digest[:])
}

// weightSources are the digests of the weight files at the top of a model's source directory, each named under
// weightSourcePrefix: what RecordModelConfig records beside the config files, read once where a model is converted or
// registered, never where the catalog is walked.
func weightSources(directory string) ([]ConfigSource, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	var sources []ConfigSource
	for _, entry := range entries {
		if entry.IsDir() || !strings.EqualFold(filepath.Ext(entry.Name()), ".safetensors") {
			continue
		}
		digest, err := fileSHA256(filepath.Join(directory, entry.Name()))
		if err != nil {
			return nil, err
		}
		sources = append(sources, ConfigSource{Name: weightSourcePrefix + entry.Name(), SHA256: digest})
	}
	return sources, nil
}

// fileSHA256 is a file's digest, read whole.
func fileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

// ConfigSources is each model's recorded sources: every file the store's
// model configs for it read, from the directory the model was converted or
// registered from, each with the digest its newest record gave it.
func ConfigSources(ctx context.Context, store overgodb.DocumentReader) (map[artifact.ID][]ConfigSource, error) {
	sources := map[artifact.ID][]ConfigSource{}
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindProfile, MediaType: ModelConfigMediaType, Schema: ModelConfigSchema}},
		Order:     overgodb.DocumentNewestFirst,
	}, ParseModelConfigDocument, func(_ overgodb.DocumentView, document ModelConfigDocument) error {
		for _, source := range document.Sources {
			if !slices.ContainsFunc(sources[document.Model], func(kept ConfigSource) bool { return kept.Name == source.Name }) {
				sources[document.Model] = append(sources[document.Model], source)
			}
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
	// The source weights are recorded beside the config files, so a model takes another's place only on the same
	// weights (WeightsKey).
	weights, err := weightSources(source)
	if err != nil {
		return artifact.ID{}, false, err
	}
	sources = append(sources, weights...)
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
