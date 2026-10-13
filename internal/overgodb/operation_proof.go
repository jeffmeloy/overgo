package overgodb

import (
	"context"
	"encoding/json"

	"overgo/internal/artifact"
)

// OperationProofAlias names the newest proof a store carries.
const OperationProofAlias = "store-operation/proof"

// operationProofKind prefixes the proof's schema across versions. It is a
// guarded kind: the store admits it only from the command that ran the
// checks, so a proof is never a document someone typed.
const operationProofKind = "overgo/store-operation-proof/"

var operationProofContract = artifact.DocumentContract{
	Kind:      artifact.KindEvidence,
	MediaType: "application/vnd.overgo.store-operation-proof+json",
	Schema:    operationProofKind + "v1",
}

// ConsumerCheck is one store consumer's verdict on a candidate.
type ConsumerCheck struct {
	Name   string `json:"name"`
	Passed bool   `json:"passed"`
	Detail string `json:"detail,omitzero"`
}

// OperationProof is what a store maintenance operation did to a candidate
// and what the store's consumers said of the result, committed into that
// candidate before it may go into service.
type OperationProof struct {
	Backup         string          `json:"backup"`
	HeadBefore     string          `json:"head_before"`
	SequenceBefore uint64          `json:"sequence_before"`
	HeadAfter      string          `json:"head_after"`
	SequenceAfter  uint64          `json:"sequence_after"`
	LinkedBlobs    int             `json:"linked_blobs"`
	Roots          int             `json:"roots"`
	RootsMissing   int             `json:"roots_missing"`
	Retained       int             `json:"retained"`
	Unreachable    int             `json:"unreachable"`
	Released       int             `json:"released"`
	ReleasedBytes  int64           `json:"released_bytes"`
	BlobsRemoved   int             `json:"blobs_removed"`
	Pack           PackReport      `json:"pack,omitzero"`
	Checks         []ConsumerCheck `json:"checks"`
	Passed         bool            `json:"passed"`
}

// Verdict settles Passed: every check passed, and there was one.
func (p *OperationProof) Verdict() bool {
	p.Passed = len(p.Checks) != 0
	for _, check := range p.Checks {
		p.Passed = p.Passed && check.Passed
	}
	return p.Passed
}

// Batch publishes the proof and moves the alias onto it by compare-and-set.
func (p OperationProof) Batch(ctx context.Context, store *Store) (artifact.Batch, error) {
	data, err := json.Marshal(p)
	if err != nil {
		return artifact.Batch{}, err
	}
	content, err := operationProofContract.ContentBytes(data)
	if err != nil {
		return artifact.Batch{}, err
	}
	binding := artifact.AliasBinding{Name: OperationProofAlias, Target: content.Descriptor.ID}
	if previous, found, err := artifact.ResolveAlias(ctx, store, OperationProofAlias); err != nil {
		return artifact.Batch{}, err
	} else if found {
		binding = artifact.AliasMove(binding.Name, binding.Target, previous)
	}
	return artifact.Batch{
		Key:      "store-operation/proof/" + content.Descriptor.ID.String(),
		Contents: []artifact.Content{content},
		Aliases:  []artifact.AliasBinding{binding},
	}, nil
}
