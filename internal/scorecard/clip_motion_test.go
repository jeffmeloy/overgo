package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestAClipRecordsItsFrameToFrameSimilarity holds a video model's quality cell to the motion its reading recorded: the
// mean and lowest similarity of a frame to the next over its clips, and the share below their source clips', stated
// beside the judged share, which stays the cell's value; a reading of pictures states none.
func TestAClipRecordsItsFrameToFrameSimilarity(t *testing.T) {
	t.Parallel()
	summary := evaluation.EvalSummary{
		Suite: "vidgen", Prompting: "judged", Record: testutil.ArtifactID(t, artifact.KindEvaluation, "motion"),
		Metrics: map[string]float64{
			"accuracy": 0.75, evaluation.SoftMetric: 0.61,
			evaluation.FrameSimilarityMetric: 0.9294, evaluation.LowestFrameSimilarityMetric: 0.514, evaluation.BelowSourceMetric: 0.25,
		},
		Directions: map[string]runrecord.Direction{
			"accuracy": runrecord.DirectionMaximize, evaluation.SoftMetric: runrecord.DirectionMaximize,
			evaluation.FrameSimilarityMetric: runrecord.DirectionMaximize, evaluation.LowestFrameSimilarityMetric: runrecord.DirectionMaximize,
			evaluation.BelowSourceMetric: runrecord.DirectionMinimize,
		},
	}
	suite := map[string]evaluation.SuiteDescriptor{"vidgen": {Kind: "generated-answer", Source: "vidgen", Cases: 4}}
	cell, _ := qualityCell(recipe.TaskVideoGen, []evaluation.EvalSummary{summary}, suite)
	if cell.Value != 0.75 || cell.Motion == nil || cell.Motion.Mean != 0.9294 || cell.Motion.Lowest != 0.514 || cell.Motion.BelowSource == nil || *cell.Motion.BelowSource != 0.25 ||
		!strings.Contains(cell.QualityText(), "frame to frame 0.9294, lowest 0.514, 0.25 of clips below their source's lowest") {
		t.Fatalf("a reading of clips reads %v with motion %+v as %q, want 0.75 with its motion beside it", cell.Value, cell.Motion, cell.QualityText())
	}
	t.Logf("the video cell reads %q", cell.QualityText())
	for _, metric := range []string{evaluation.FrameSimilarityMetric, evaluation.LowestFrameSimilarityMetric, evaluation.BelowSourceMetric} {
		delete(summary.Metrics, metric)
	}
	if pictures, _ := qualityCell(recipe.TaskImageGen, []evaluation.EvalSummary{summary}, suite); pictures.Motion != nil || strings.Contains(pictures.QualityText(), "frame to frame") {
		t.Fatalf("a reading with no motion reads %q", pictures.QualityText())
	}
}
