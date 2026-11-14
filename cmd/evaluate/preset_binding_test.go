package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAPresetReadingReadsOnlyItsRendersJudgments holds a preset reading's records to the renders they judged
// (SQA-379, judgedRender): a record is read only for the media its render drew, each its content, request and producing
// run, and only as judged by the reading's one judge and scored by its one scorer.
func TestAPresetReadingReadsOnlyItsRendersJudgments(t *testing.T) {
	t.Parallel()
	drawn := renderManifest{Draws: map[string]renderedMedia{"draw": {
		Input: testutil.ArtifactID(t, artifact.KindFile, "asked"), Output: testutil.ArtifactID(t, artifact.KindOutput, "drawn"),
		Run: testutil.ArtifactID(t, artifact.KindRun, "drew"),
	}}}
	judged := evaluation.MediaInput{
		Content: drawn.Draws["draw"].Output, Request: drawn.Draws["draw"].Input, Producer: drawn.Draws["draw"].Run,
		Runtime: evaluation.MediaRuntime{Model: testutil.ArtifactID(t, artifact.KindModel, "judge"), Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "judging")},
	}
	record, scorer := testutil.ArtifactID(t, artifact.KindEvaluation, "judged"), testutil.ArtifactID(t, artifact.KindFile, "scorer")
	var judge judging
	testcheck.NoError(t, judgedRender(record, []evaluation.MediaInput{judged}, scorer, drawn, &judge))
	other := judged
	other.Content = testutil.ArtifactID(t, artifact.KindOutput, "drawn elsewhere")
	if err := judgedRender(record, []evaluation.MediaInput{other}, scorer, drawn, &judge); err == nil {
		t.Error("a record that judged media its render did not draw was read")
	}
	other = judged
	other.Runtime.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "another judge")
	if err := judgedRender(record, []evaluation.MediaInput{other}, scorer, drawn, &judge); err == nil {
		t.Error("an arm judged by another judge than the reading's was read")
	}
	if err := judgedRender(record, []evaluation.MediaInput{judged}, testutil.ArtifactID(t, artifact.KindFile, "another scorer"), drawn, &judge); err == nil {
		t.Error("an arm scored by another scorer than the reading's was read")
	}
}
