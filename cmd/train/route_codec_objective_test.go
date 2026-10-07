package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// pictureObjectiveAlias names the picture objective a test registers in its
// own store.
const pictureObjectiveAlias = "objective.registered.pictures"

// registerPictureObjective writes count random PNG pictures, side by side,
// registers them as a dataset split in halves, and registers over its
// training membership a picture objective judged by the peak signal-to-noise
// ratio of pictures in [0, 1]. It returns the objective and a weights file to
// record.
func registerPictureObjective(t *testing.T, ctx context.Context, store *overgodb.Store, root string, count, side int) (trainingprogram.ObjectiveDocument, string) {
	t.Helper()
	corpus := t.TempDir()
	rng := rand.New(rand.NewPCG(7, 7))
	for index := range count {
		picture := image.NewNRGBA(image.Rect(0, 0, side, side))
		for y := range side {
			for x := range side {
				picture.SetNRGBA(x, y, color.NRGBA{R: uint8(rng.UintN(256)), G: uint8(rng.UintN(256)), B: uint8(rng.UintN(256)), A: math.MaxUint8})
			}
		}
		var encoded bytes.Buffer
		if err := png.Encode(&encoded, picture); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("picture-%04d.png", index)), encoded.Bytes(), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	source, training := registerSplit(t, ctx, store, "pictures", corpus, count, nil)
	return registerObjective(t, ctx, store, root, pictureObjectiveAlias, trainingprogram.ObjectiveSpec{
		Name: "pictures", Kind: trainingprogram.ObjectiveImageLatent,
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityImage}, Outputs: []recipecontract.Modality{recipecontract.ModalityImage}},
		Dataset:   source, Split: training, Metric: trainingprogram.MetricImagePSNRUnit,
	})
}

// TestCodecAdaptationPublishesItsHeldOutVerdict walks a registered dataset
// of pictures through a picture objective the way the codec route does: its
// examples come through the training data path keyed by record, each
// picture its own target in [0, 1] with the digest its tokens are kept
// under; a picture of another size is refused; and a held-out run whose
// trained model draws the held-out pictures closer than the base model did
// publishes a verdict that measures both by peak signal-to-noise ratio, on
// which the objective climbs from declared to approved.
func TestCodecAdaptationPublishesItsHeldOutVerdict(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const pictures, side = 12, 8
	objective, weights := registerPictureObjective(t, ctx, store, root, pictures, side)

	data, digests, err := codecPictures(ctx, store, objective, side)
	if err != nil || !data.Training.Valid() || !data.Heldout.Valid() {
		t.Fatalf("objective pictures = (streams %s and %s, %v), want both memberships' streams", data.Training, data.Heldout, err)
	}
	if len(data.Examples) != pictures || len(digests) != pictures {
		t.Fatalf("read %d pictures and %d digests, want all %d", len(data.Examples), len(digests), pictures)
	}
	kept := map[string]bool{}
	for record, example := range data.Examples {
		if len(example.Target) != 3*side*side || kept[digests[record]] {
			t.Fatalf("record %s holds %d values under digest %q, want one picture of %d under a digest of its own", record, len(example.Target), digests[record], 3*side*side)
		}
		kept[digests[record]] = true
		for _, value := range example.Target {
			if value < 0 || value > 1 {
				t.Fatalf("record %s holds %v, outside [0, 1]", record, value)
			}
		}
	}
	if _, _, err := codecPictures(ctx, store, objective, side+1); err == nil {
		t.Fatal("pictures were read for a pair that draws another size")
	}

	// A base model draws every value a tenth off, 20 dB; the trained one a
	// hundredth off, 40 dB.
	const baseError, trainedError = 0.1, 0.01
	off := baseError
	result, err := trainingworkflow.RunHeldout(ctx, store, trainingworkflow.HeldoutRun{
		Alias: pictureObjectiveAlias, ModelPath: weights, Data: data, Host: true,
		Train: func(ctx context.Context, recipeID artifact.ID, shown []trainingworkflow.HeldoutExample) (trainingprogram.Checkpoint, error) {
			if len(shown) == 0 || len(shown) >= pictures {
				t.Errorf("training saw %d of %d pictures, want the training membership alone", len(shown), pictures)
			}
			config, err := trainingprogram.BuiltinOptimizerPolicy().Config(testMatrix.End)
			if err != nil {
				return trainingprogram.Checkpoint{}, err
			}
			matrix := newHostMatrix(t, trainingprogram.ObjectiveImageLatent, config)
			step := matrix.step()
			off = trainedError
			return publishPlannedTrained(ctx, store, recipeID, filepath.Join(root, "checkpoint"), matrix, trainingdata.StreamState{Identity: data.Training, Position: uint64(step.Step)})
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			drawn := make([]float64, len(example.Target))
			for index, value := range example.Target {
				drawn[index] = value + off
			}
			return trainingworkflow.HeldoutPrediction{Values: drawn}, nil
		},
	})
	if err != nil || !result.Passed {
		t.Fatalf("held-out run = (passed=%t, %v), want a passed verdict", result.Passed, err)
	}
	var verdict struct {
		Metric         trainingprogram.EvaluationMetric `json:"metric"`
		BaselineValue  float64                          `json:"baseline_value"`
		CandidateValue float64                          `json:"candidate_value"`
	}
	_, content, found, err := store.OpenContent(ctx, result.Verdict)
	if err != nil || !found {
		t.Fatalf("verdict %s content = (found %t, %v)", result.Verdict, found, err)
	}
	document, err := io.ReadAll(content)
	if err != nil {
		t.Fatal(err)
	}
	const tolerance = 1e-3
	if err := json.Unmarshal(document, &verdict); err != nil || verdict.Metric != trainingprogram.MetricImagePSNRUnit ||
		math.Abs(verdict.BaselineValue-20) > tolerance || math.Abs(verdict.CandidateValue-40) > tolerance {
		t.Fatalf("verdict = %+v (%v), want %s from 20 dB to 40 dB", verdict, err, trainingprogram.MetricImagePSNRUnit)
	}
	if _, err := trainingworkflow.PromoteObjectiveAdaptive(ctx, store, pictureObjectiveAlias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := trainingworkflow.PromoteObjectiveApproved(ctx, store, pictureObjectiveAlias, []artifact.ID{result.Verdict})
	if err != nil || approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("approval = (%q, %v), want approved on the held-out verdict", approved.Authority, err)
	}
}
