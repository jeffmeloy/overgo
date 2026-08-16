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
	roadmapCompletionVersion   uint16 = 1
	roadmapCompletionMediaType        = "application/vnd.overgo.roadmap-completion+json"
	roadmapCompletionSchema           = "overgo/roadmap-completion/v1"
)

var roadmapVerifierOutputContract = artifact.DocumentContract{
	Kind: artifact.KindOutput, MediaType: "application/x-ndjson", Schema: "overgo/go-test-json/v1",
}

// RoadmapCompletion binds a historically landed row to a passing exact
// verifier run at one reachable commit.
type RoadmapCompletion struct {
	Version    uint16      `json:"version"`
	Row        string      `json:"row"`
	Verifier   string      `json:"verifier"`
	CodeCommit string      `json:"code_commit"`
	Run        artifact.ID `json:"run"`
	ID         artifact.ID `json:"-"`
}

var roadmapCompletionCodec = artifact.JSONDocumentCodec("roadmap completion", artifact.KindEvidence,
	roadmapCompletionMediaType, roadmapCompletionSchema, canonicalizeRoadmapCompletion,
	func(value RoadmapCompletion) artifact.ID { return value.ID },
	func(value *RoadmapCompletion, id artifact.ID) { value.ID = id }, nil)

// ExecuteRoadmapCompletion records a passing exact verifier without inventing
// gate history for work that predates typed gate acceptance evidence.
func ExecuteRoadmapCompletion(ctx context.Context, root string, repository artifact.Repository, row, verifier string) (RoadmapCompletion, error) {
	execution, err := executeRoadmapVerifier(root, verifier)
	if err != nil {
		return RoadmapCompletion{}, err
	}
	if err := testevidence.VerifyGoTestTarget(verifier, execution.Output); err != nil {
		return RoadmapCompletion{}, fmt.Errorf("roadmap completion is not passing exact evidence: %w", err)
	}
	output, err := roadmapVerifierOutputContract.ContentBytes([]byte(execution.Output))
	if err != nil {
		return RoadmapCompletion{}, err
	}
	recipe, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("overgo-roadmap-completion/v1\x00"+row+"\x00"+verifier))
	if err != nil {
		return RoadmapCompletion{}, err
	}
	run, err := runrecord.NewBoundRun(recipe, runrecord.OutcomeSucceeded, nil, []artifact.ID{output.Descriptor.ID}, "",
		execution.CodeCommit, execution.Environment, execution.MeasuredNS,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseTest, DurationNS: execution.MeasuredNS}})
	if err != nil {
		return RoadmapCompletion{}, err
	}
	completion, err := roadmapCompletionCodec.New(RoadmapCompletion{
		Version: roadmapCompletionVersion, Row: row, Verifier: verifier,
		CodeCommit: execution.CodeCommit, Run: run.ID,
	})
	if err != nil {
		return RoadmapCompletion{}, err
	}
	completionContent, err := roadmapCompletionCodec.Content(completion)
	if err != nil {
		return RoadmapCompletion{}, err
	}
	batch, err := run.Batch("automation/roadmap-completion/" + completion.ID.String())
	if err != nil {
		return RoadmapCompletion{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: recipe})
	batch.Contents = append(batch.Contents, execution.EnvironmentContent, output, completionContent)
	batch.Lineage = append(batch.Lineage, artifact.Lineage{
		Child: completion.ID, Parent: run.ID, Relation: artifact.RelationDependsOn,
	})
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return RoadmapCompletion{}, err
	}
	return completion, nil
}

// ReadRoadmapCompletion returns false for unrelated evidence.
func ReadRoadmapCompletion(ctx context.Context, reader artifact.Reader, id artifact.ID) (RoadmapCompletion, bool, error) {
	return readTypedDocument(ctx, reader, id, roadmapCompletionCodec.Contract, roadmapCompletionCodec.Read)
}

func canonicalizeRoadmapCompletion(value *RoadmapCompletion) error {
	if value == nil || value.Version != roadmapCompletionVersion || !textcheck.LowerIdentifier(value.Row, 2048) ||
		!validCommit(value.CodeCommit) || value.Run.Kind() != artifact.KindRun {
		return errors.New("plan: invalid roadmap completion")
	}
	if err := testevidence.ValidateGoTestCommand(value.Verifier); err != nil {
		return fmt.Errorf("plan: invalid roadmap completion verifier: %w", err)
	}
	return nil
}
