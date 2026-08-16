package plan

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/textcheck"
)

const (
	roadmapProbeMediaType = "application/vnd.overgo.roadmap-probe+json"
	roadmapProbeSchema    = "overgo/roadmap-probe/v1"
)

// RoadmapProbe binds one isolated expected failure to a roadmap verifier.
type RoadmapProbe struct {
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

func identifyRoadmapProbe(value RoadmapProbe) (RoadmapProbe, artifact.Content, error) {
	identified, err := roadmapProbeCodec.New(value)
	if err != nil {
		return RoadmapProbe{}, artifact.Content{}, err
	}
	content, err := roadmapProbeCodec.Content(identified)
	if err != nil {
		return RoadmapProbe{}, artifact.Content{}, err
	}
	return identified, content, nil
}

// ReadRoadmapProbe returns false when id is not roadmap-probe evidence.
func ReadRoadmapProbe(ctx context.Context, reader artifact.Reader, id artifact.ID) (RoadmapProbe, bool, error) {
	return readTypedDocument(ctx, reader, id, roadmapProbeCodec.Contract, roadmapProbeCodec.Read)
}

// ExecuteRoadmapProbe accepts only a healthy verifier run whose named target
// is absent, then binds that observation to immutable run evidence and HEAD.
func ExecuteRoadmapProbe(ctx context.Context, root string, repository artifact.Repository, row, verifier string) (RoadmapProbe, error) {
	execution, err := executeRoadmapVerifier(root, verifier)
	if err != nil {
		return RoadmapProbe{}, err
	}
	if err := testevidence.VerifyGoTestTargetAbsent(verifier, execution.Output); err != nil {
		return RoadmapProbe{}, fmt.Errorf("roadmap probe is not an isolated absent target: %w", err)
	}
	recipe, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("overgo-roadmap-probe/v1"))
	if err != nil {
		return RoadmapProbe{}, err
	}
	run, err := runrecord.NewBoundRun(recipe, runrecord.OutcomeFailed, nil, nil, "target-absent",
		execution.CodeCommit, execution.Environment,
		execution.MeasuredNS, []runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: execution.MeasuredNS}})
	if err != nil {
		return RoadmapProbe{}, err
	}
	batch, err := run.Batch("automation/roadmap-probe-run/" + run.ID.String())
	if err != nil {
		return RoadmapProbe{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipe})
	probe, probeContent, err := identifyRoadmapProbe(RoadmapProbe{
		Row: row, Verifier: verifier, CodeCommit: execution.CodeCommit, Run: run.ID, ExpectedFailure: "target-absent",
	})
	if err != nil {
		return RoadmapProbe{}, err
	}
	batch.Contents = append(batch.Contents, execution.EnvironmentContent, probeContent)
	batch.Lineage = append(batch.Lineage, artifact.Lineage{
		Child: probe.ID, Parent: run.ID, Relation: artifact.RelationDependsOn,
	})
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return RoadmapProbe{}, err
	}
	return probe, nil
}

func canonicalizeRoadmapProbe(value *RoadmapProbe) error {
	if value == nil || !textcheck.LowerIdentifier(value.Row, 2048) ||
		!validCommit(value.CodeCommit) || value.Run.Kind() != artifact.KindRun ||
		!textcheck.LowerIdentifier(value.ExpectedFailure, 2048) {
		return errors.New("plan: invalid roadmap probe")
	}
	if err := testevidence.ValidateGoTestCommand(value.Verifier); err != nil {
		return fmt.Errorf("plan: invalid roadmap probe verifier: %w", err)
	}
	return nil
}
