package main

import (
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// syntheticCapture writes a capture of samples in groups of five: an aligned source of width 16 whose last column is a
// constant, as a normalization's output carries one, a target of width 8 that is a fixed map of it plus a large shared
// mean and noise at the given share, and an independent source of the same width. Its metadata names a residual no
// measurement gave.
func syntheticCapture(t *testing.T, samples int, noise float64) string {
	t.Helper()
	const sourceWidth, targetWidth, groupSize, targetMean = 16, 8, 5, 40.0
	draws := rand.New(rand.NewPCG(7, 11))
	mapping := make([]float64, sourceWidth*targetWidth)
	for index := range mapping {
		mapping[index] = draws.NormFloat64()
	}
	aligned, independent := make([]float32, samples*sourceWidth), make([]float32, samples*sourceWidth)
	target, groups := make([]float32, samples*targetWidth), make([]float32, samples)
	for sample := range samples {
		groups[sample] = float32(sample / groupSize)
		for column := range sourceWidth {
			aligned[sample*sourceWidth+column] = float32(draws.NormFloat64())
			independent[sample*sourceWidth+column] = float32(draws.NormFloat64())
		}
		aligned[sample*sourceWidth+sourceWidth-1] = 1
		for column := range targetWidth {
			value := targetMean
			for index := range sourceWidth - 1 {
				value += float64(aligned[sample*sourceWidth+index]) * mapping[index*targetWidth+column]
			}
			target[sample*targetWidth+column] = float32(value + noise*draws.NormFloat64())
		}
	}
	path := filepath.Join(t.TempDir(), "capture.safetensors")
	testcheck.NoError(t, safetensors.Save(path,
		map[string][]float32{targetTensor: target, groupsTensor: groups, sourcePrefix + "aligned": aligned, sourcePrefix + "independent": independent},
		map[string][]int{targetTensor: {samples, targetWidth}, groupsTensor: {samples}, sourcePrefix + "aligned": {samples, sourceWidth}, sourcePrefix + "independent": {samples, sourceWidth}},
		map[string]string{"residual": "0.123456"}))
	return path
}

// TestSeamScoreComesFromMeasuredRepresentations holds seam-align's scores to the captured representations: a capture's
// ranking is its held-out residuals recomputed from its tensors, whatever its metadata says, and moves with their
// noise; a seam that holds a linear relation ranks first on the linear rung, while an independent source, and the
// aligned source paired across groups, read near one; the in-sample residual of that unrelated pairing, by contrast,
// reads low, since the target's shared mean and the source's constant explain it. A driven composite's score is the
// held-out residual of its recorded activations against the linear rung's bound, so the aligned seam is fit and a
// planted one is not, and a run specification that types a score in is refused.
func TestSeamScoreComesFromMeasuredRepresentations(t *testing.T) {
	t.Parallel()
	const samples, linearNoise, loudNoise, unrelatedFloor = 320, 0.5, 4.0, 0.9
	path := syntheticCapture(t, samples, linearNoise)
	ranks := testcheck.Must(rankCapture(path))(t)
	if len(ranks) != 2 || ranks[0].Source != "aligned" || ranks[0].Rung != composition.AdapterLinear || !(ranks[0].HeldOut < ranks[0].Unrelated) {
		t.Fatalf("the aligned seam does not rank first on the linear rung below its unrelated pairing: %+v", ranks)
	}
	capture := testcheck.Must(safetensors.OpenFile(path))(t)
	defer capture.Close()
	source, target := testcheck.Must(captureRows(capture, sourcePrefix+"aligned"))(t), testcheck.Must(captureRows(capture, targetTensor))(t)
	groupRows := testcheck.Must(captureRows(capture, groupsTensor))(t)
	groups := make([]int, len(groupRows))
	for index, row := range groupRows {
		groups[index] = int(row[0])
	}
	if recomputed := testcheck.Must(composition.HeldOutSeamResidual(source, target, groups, seamFolds))(t); ranks[0].HeldOut != recomputed {
		t.Errorf("the ranking reports %v where the tensors give %v", ranks[0].HeldOut, recomputed)
	}
	if loud := testcheck.Must(rankCapture(syntheticCapture(t, samples, loudNoise)))(t); !(loud[0].HeldOut > ranks[0].HeldOut) {
		t.Errorf("louder noise left the aligned seam's residual at %v from %v", loud[0].HeldOut, ranks[0].HeldOut)
	}
	if independent := ranks[1]; independent.HeldOut < unrelatedFloor || independent.Rung != composition.AdapterBridge || ranks[0].Unrelated < unrelatedFloor {
		t.Errorf("an unrelated seam reads below %v held out: independent %+v, aligned paired across groups %v", unrelatedFloor, independent, ranks[0].Unrelated)
	}
	pairing := testcheck.Must(composition.UnrelatedPairing(groups))(t)
	paired := make([][]float64, len(target))
	for sample, other := range pairing {
		paired[sample] = target[other]
	}
	if inSample := testcheck.Must(composition.SeamAlignmentResidual(source, paired))(t); inSample >= math.Sqrt(0.5) {
		t.Errorf("the unrelated pairing's in-sample residual reads %v; the held-out measure guards nothing here", inSample)
	}
	verdicts := make(map[string]bool)
	for name, activations := range map[string][][]float64{"aligned": source, "planted": testcheck.Must(captureRows(capture, sourcePrefix+"independent"))(t)} {
		dimension := testcheck.Must(seamDimension(activations, target))(t)
		judged := testcheck.Must(composition.ScoreCompositeSelection([]composition.CompositeScore{{
			Composite: testutil.ArtifactID(t, artifact.KindModel, name), Evaluation: testutil.ArtifactID(t, artifact.KindEvidence, name),
			Dimensions: map[string]composition.CompositeDimension{seamDimensionName: dimension},
		}}))(t)
		verdicts[name] = judged[0].Fit
	}
	if !verdicts["aligned"] || verdicts["planted"] {
		t.Errorf("the measured seam scores fit %v and the planted one %v, want the aligned fit alone", verdicts["aligned"], verdicts["planted"])
	}
	var typed driveTargetSpecification
	if err := strictjson.DecodeBytes([]byte(`{"label": "seam", "score": {"dimensions": {}}}`), &typed); err == nil {
		t.Error("a run specification that types its score in was accepted")
	}
}
