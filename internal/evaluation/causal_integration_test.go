package evaluation

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestRSICausalChainClosure pins the evaluation side of the chain: an
// evidence bundle carries the causal context of the execution that
// produced it, answering to the same root as the attempt it evaluates,
// and a causal binding whose fields disagree refuses.
func TestRSICausalChainClosure(t *testing.T) {
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	proposal := id(artifact.KindEvidence, "evaluation-proposal")
	motivation := id(artifact.KindEvidence, "evaluation-motivation")
	root, err := runrecord.NewCausalRoot(runrecord.TriggerControllerProposal, proposal, motivation)
	if err != nil {
		t.Fatal(err)
	}
	value := EvaluationEvidence{
		Version: artifact.InitialDocumentVersion,
		Plan:    id(artifact.KindProfile, "plan"), Acceptance: id(artifact.KindProfile, "acceptance"),
		Evaluator: id(artifact.KindEvidence, "evaluator"), Report: id(artifact.KindEvaluation, "report"),
		Run: id(artifact.KindRun, "run"), Evaluation: id(artifact.KindEvaluation, "evaluation"),
		ModelDefinition: id(artifact.KindModelDefinition, "definition"), Recipe: id(artifact.KindRecipe, "recipe"),
		Dataset: id(artifact.KindDataset, "dataset"), Split: id(artifact.KindDatasetShard, "split"),
		Environment: id(artifact.KindEvidence, "environment"), CodeCommit: strings.Repeat("ef", 20),
		Phases:  []runrecord.PhaseMetric{{Phase: runrecord.PhaseDecode, DurationNS: 3}},
		Metrics: []runrecord.Metric{{Name: "accuracy", Value: 0.5, Direction: runrecord.DirectionMaximize}},
		Causal:  &root,
	}
	evidence, err := evaluationEvidenceCodec.New(value)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Causal == nil || evidence.Causal.Root != proposal ||
		len(evidence.Causal.Motivation) != 1 || evidence.Causal.Motivation[0] != motivation {
		t.Fatalf("evidence left the causal chain: %+v", evidence.Causal)
	}
	parsed, err := ParseEvaluationEvidence(mustContent(t, evidence).Data)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Causal == nil || parsed.Causal.Root != proposal {
		t.Fatalf("parsed evidence lost the causal root: %+v", parsed.Causal)
	}

	broken := root
	broken.RecoveredFrom = id(artifact.KindEvidence, "not-my-trigger")
	value.Causal = &broken
	if _, err := evaluationEvidenceCodec.New(value); err == nil {
		t.Fatal("evidence accepted a broken causal binding")
	}
}

func mustContent(t *testing.T, evidence EvaluationEvidence) artifact.Content {
	t.Helper()
	content, err := evidence.Content()
	if err != nil {
		t.Fatal(err)
	}
	return content
}
