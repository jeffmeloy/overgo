package codemanifest

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
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

// digestContract types the durable footprint of one derived manifest:
// identity, source, and scale, without the derived content. A manifest
// is recomputable from its git source, so the store keeps the claim,
// not the cache -- persisting full manifests per gate run was the
// store's entire growth curve (6.07 GB across 146 documents against a
// 450 MB knowledge base, measured 2026-08-27).
var digestContract = artifact.JSONContract(artifact.KindEvidence, "overgo.code-manifest-digest.v1")

// DigestAlias names the digest record for one manifest identity.
func DigestAlias(manifest artifact.ID) string {
	return "code-manifest/digest/" + manifest.String()
}

// PublishDigest records the manifest's summary and aliases it by the
// manifest identity; the full content stays derivable, never stored.
func PublishDigest(ctx context.Context, repository artifact.Repository, manifest Manifest) (artifact.CommitID, error) {
	if ctx == nil || repository == nil {
		return artifact.CommitID{}, errors.New("code manifest: nil publication context or repository")
	}
	summary, err := Summarize(manifest)
	if err != nil {
		return artifact.CommitID{}, err
	}
	content, err := artifact.JSONContent(digestContract, summary)
	if err != nil {
		return artifact.CommitID{}, err
	}
	alias := DigestAlias(manifest.ID)
	batch := artifact.Batch{
		Key:      "code-manifest-digest/" + manifest.ID.String(),
		Contents: []artifact.Content{content},
		// The manifest identity stays a registered artifact so gate
		// lineage that names it as a parent remains resolvable; only
		// its derived content is no longer stored.
		Artifacts: []artifact.Descriptor{{ID: manifest.ID}},
		Aliases:   []artifact.AliasBinding{{Name: alias, Target: content.Descriptor.ID}},
	}
	commit, err := repository.Commit(ctx, batch)
	if errors.Is(err, artifact.ErrNoChange) {
		return commit, nil
	}
	return commit, err
}

// LoadDigest reads the digest record for one manifest identity.
func LoadDigest(ctx context.Context, repository artifact.Reader, manifest artifact.ID) (Summary, error) {
	id, found, err := artifact.ResolveAlias(ctx, repository, DigestAlias(manifest))
	if err != nil {
		return Summary{}, err
	}
	if !found {
		return Summary{}, fmt.Errorf("code manifest: no digest recorded for %s", manifest)
	}
	content, found, err := artifact.ReadContent(ctx, repository, id)
	if err != nil || !found {
		return Summary{}, fmt.Errorf("code manifest: digest content for %s is unreadable", manifest)
	}
	var summary Summary
	if err := strictjson.DecodeBytes(content.Data, &summary); err != nil {
		return Summary{}, err
	}
	return summary, nil
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
