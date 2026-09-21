package overgodb

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"

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

// LastRelease reports the sequence of the newest commit that released
// content, zero when none has: the chain's own record that the store was
// mutated, which no operator has to remember to declare.
func (s *Store) LastRelease() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var last uint64
	for _, locator := range s.state.contents.locators {
		last = max(last, locator.released)
	}
	return last
}

// RequireProvenSince refuses a store whose newest release after sequence is
// not followed by a passed proof: the mutation went into service without the
// candidate checks its class owns.
func (s *Store) RequireProvenSince(ctx context.Context, sequence uint64) error {
	released := s.LastRelease()
	if released <= sequence {
		return nil
	}
	refusal := fmt.Errorf("overgodb: content was released at sequence %d with no passed proof after it; prove the store on a candidate with `go run ./cmd/store-precheck -backup <sealed backup> -swap`", released)
	id, found, err := artifact.ResolveAlias(ctx, s, OperationProofAlias)
	if err != nil || !found {
		return cmp.Or(err, refusal)
	}
	introduced, _, err := s.ArtifactIntroduction(ctx, id)
	if err != nil {
		return err
	}
	content, _, err := artifact.ReadContent(ctx, s, id)
	if err != nil {
		return err
	}
	var proof OperationProof
	if err := json.Unmarshal(content.Data, &proof); err != nil {
		return err
	}
	if introduced.Sequence < released || !proof.Passed {
		return refusal
	}
	return nil
}
