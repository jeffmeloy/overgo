package plan

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/testevidence"
	"overgo/internal/textcheck"
)

const (
	roadmapProbeVersion   uint16 = 1
	roadmapProbeMediaType        = "application/vnd.overgo.roadmap-probe+json"
	roadmapProbeSchema           = "overgo/roadmap-probe/v1"
)

// RoadmapProbe binds one isolated expected failure to a roadmap verifier.
type RoadmapProbe struct {
	Version         uint16      `json:"version"`
	Row             string      `json:"row"`
	Verifier        string      `json:"verifier"`
	CodeCommit      string      `json:"code_commit"`
	Run             artifact.ID `json:"run"`
	ExpectedFailure string      `json:"expected_failure"`
	ID              artifact.ID `json:"-"`
}

var roadmapProbeCodec = artifact.JSONDocumentCodec("roadmap probe", artifact.KindEvidence, roadmapProbeMediaType, roadmapProbeSchema,
	canonicalizeRoadmapProbe, func(value RoadmapProbe) artifact.ID { return value.ID },
	func(value *RoadmapProbe, id artifact.ID) { value.ID = id }, nil)

// RecordRoadmapProbe stores immutable probe evidence; it grants no dispatch or
// landing authority.
func RecordRoadmapProbe(ctx context.Context, repository artifact.Repository, value RoadmapProbe) (RoadmapProbe, error) {
	value.Version = roadmapProbeVersion
	identified, err := roadmapProbeCodec.New(value)
	if err != nil {
		return RoadmapProbe{}, err
	}
	content, err := roadmapProbeCodec.Content(identified)
	if err != nil {
		return RoadmapProbe{}, err
	}
	batch, err := artifact.NewDocumentBatch("automation/roadmap-probe/"+identified.ID.String(), []artifact.Content{content},
		[]artifact.Lineage{{Child: identified.ID, Parent: identified.Run, Relation: artifact.RelationDependsOn}}, nil)
	if err != nil {
		return RoadmapProbe{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return identified, err
}

// ReadRoadmapProbe returns false when id is not roadmap-probe evidence.
func ReadRoadmapProbe(ctx context.Context, reader artifact.Reader, id artifact.ID) (RoadmapProbe, bool, error) {
	return readTypedDocument(ctx, reader, id, roadmapProbeCodec.Contract, roadmapProbeCodec.Read)
}

func canonicalizeRoadmapProbe(value *RoadmapProbe) error {
	if value == nil || value.Version != roadmapProbeVersion || !textcheck.LowerIdentifier(value.Row, 2048) ||
		!validCommit(value.CodeCommit) || value.Run.Kind() != artifact.KindRun ||
		!textcheck.LowerIdentifier(value.ExpectedFailure, 2048) {
		return errors.New("plan: invalid roadmap probe")
	}
	if err := testevidence.ValidateGoTestCommand(value.Verifier); err != nil {
		return fmt.Errorf("plan: invalid roadmap probe verifier: %w", err)
	}
	return nil
}
