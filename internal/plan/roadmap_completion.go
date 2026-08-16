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
	roadmapCompletionVersion   uint16 = 1
	roadmapCompletionMediaType        = "application/vnd.overgo.roadmap-completion+json"
	roadmapCompletionSchema           = "overgo/roadmap-completion/v1"
)

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
