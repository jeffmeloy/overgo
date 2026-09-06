package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
)

// prepareTranscriptionSelection verifies every declared source before model
// loading. A missing split is derived from exact, target-free input identities;
// an explicit projected split must identify that same selection.
func prepareTranscriptionSelection(ctx context.Context, store artifact.Repository, suite *evaluation.TranscriptionSuite, inputs []evaluation.TranscriptionResourceInput, memoryBytes uint64) (returnErr error) {
	if ctx == nil || store == nil || suite == nil || len(suite.Cases) == 0 || len(inputs) != len(suite.Cases) {
		return errors.New("evaluate: incomplete transcription selection")
	}
	source, err := dataset.NewAudioPayloadReader(memoryBytes)
	if err != nil {
		return err
	}
	defer func() { returnErr = errors.Join(returnErr, source.Close()) }()
	byName := make(map[string]evaluation.TranscriptionResourceInput, len(inputs))
	for _, input := range inputs {
		if _, duplicate := byName[input.Name]; duplicate {
			return fmt.Errorf("evaluate: duplicate transcription input %q", input.Name)
		}
		byName[input.Name] = input
	}
	records := make([]dataset.Record, 0, len(suite.Cases))
	for _, testCase := range suite.Cases {
		input, found := byName[testCase.Name]
		if !found || input.Reference.Audio != testCase.Source.Audio {
			return fmt.Errorf("evaluate: transcription source differs for %q", testCase.Name)
		}
		if _, err := source.Read(ctx, input.Reference, input.Policy.MaximumEncodedBytes); err != nil {
			return fmt.Errorf("evaluate: transcription prerequisite %q at %q: %w", testCase.Name, input.Reference.Path, err)
		}
		coordinate, err := artifact.JSONID(artifact.KindEvidence, input.Reference.Origin)
		if err != nil {
			return err
		}
		records = append(records, dataset.Record{ID: testCase.Name + "/" + input.Reference.Audio.String() + "/" + coordinate.DigestHex(), Group: cmp.Or(testCase.Group, "all")})
	}
	// No random partitioning occurs: zero is the identity seed for an explicit
	// caller-selected membership, not a statistical sampling policy.
	membership, err := dataset.NewMembership(suite.Dataset, 0, "transcription-selected", records)
	if err != nil {
		return err
	}
	if suite.Split.Valid() && suite.Split != membership.ID {
		if suite.Prompt != "" {
			return fmt.Errorf("evaluate: projected transcription split %s differs from selected membership %s; omit split to derive it", suite.Split, membership.ID)
		}
		// Native manifests retain their already declared split protocol.
		return nil
	}
	candidate := *suite
	candidate.Split = membership.ID
	if _, err := evaluation.CompileTranscription(candidate); err != nil {
		return err
	}
	content, err := membership.Content()
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch("evaluation/transcription-selection/"+membership.ID.DigestHex(), []artifact.Content{content}, []artifact.Lineage{membership.Lineage()}, nil)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return err
	}
	suite.Split = membership.ID
	return nil
}
