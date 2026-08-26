package codemanifest

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

// Summary is a bounded inspection view that never materializes bulk records
// in command output.
type Summary struct {
	ID             artifact.ID `json:"id"`
	SourceIdentity string      `json:"source_identity"`
	Analyzer       Analyzer    `json:"analyzer"`
	BuildContexts  int         `json:"build_contexts"`
	Files          int         `json:"files"`
	Symbols        int         `json:"symbols"`
	References     int         `json:"references"`
	ExternalInputs int         `json:"external_inputs"`
	Uncertainty    int         `json:"uncertainty"`
}

// Summarize projects bounded counts and authority from a valid manifest.
func Summarize(manifest Manifest) (Summary, error) {
	if err := manifest.Validate(); err != nil {
		return Summary{}, err
	}
	return Summary{
		ID: manifest.ID, SourceIdentity: manifest.SourceIdentity, Analyzer: manifest.Analyzer,
		BuildContexts: len(manifest.BuildContexts), Files: len(manifest.Files), Symbols: len(manifest.Symbols),
		References: len(manifest.References), ExternalInputs: len(manifest.ExternalInputs), Uncertainty: len(manifest.Uncertainty),
	}, nil
}

// Publish writes one canonical manifest under an idempotent content key.
func Publish(ctx context.Context, repository artifact.Repository, manifest Manifest) (artifact.CommitID, error) {
	if ctx == nil || repository == nil {
		return artifact.CommitID{}, errors.New("code manifest: nil publication context or repository")
	}
	content, err := manifest.Content()
	if err != nil {
		return artifact.CommitID{}, err
	}
	batch, err := artifact.NewDocumentBatch("code-manifest/"+manifest.ID.String(), []artifact.Content{content}, nil, nil)
	if err != nil {
		return artifact.CommitID{}, err
	}
	return repository.Commit(ctx, batch)
}

// Load reads and strictly validates one canonical manifest.
func Load(ctx context.Context, repository artifact.Reader, id artifact.ID) (Manifest, error) {
	if id.Kind() != artifact.KindProfile {
		return Manifest{}, errors.New("code manifest: manifest identity must be a profile")
	}
	content, found, err := artifact.ReadContent(ctx, repository, id)
	if err != nil {
		return Manifest{}, err
	}
	if !found {
		return Manifest{}, fmt.Errorf("code manifest: %s not found", id)
	}
	if content.Descriptor.MediaType != MediaType || content.Descriptor.Schema != Schema {
		return Manifest{}, errors.New("code manifest: stored content has incompatible contract")
	}
	return Parse(content.Data)
}
