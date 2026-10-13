package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
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
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/rae"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
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
			return publishPlannedTrained(ctx, store, recipeID, filepath.Join(root, "checkpoint"), matrix, trainingdata.StreamState{Identity: data.Training, Position: uint64(step.Step)}, false)
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

// TestCodecProofTakesARecipeAndNoTokenCache holds what the codec route takes
// of a proof, before it reads any weight. A proof is an objective's run under
// the recipe it is given, and it keeps no tokens: it is started without a
// cache, so it reaches the store and stops only at the objective nothing
// registered there, and it is refused a cache, where every other run is
// refused without one. A proof of a manifest's frames, which no recipe is
// selected for, and a recipe that is no identity are refused by name.
func TestCodecProofTakesARecipeAndNoTokenCache(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	pair := writeBridgedPair(t, filepath.Join(root, "models"))
	cache, recipeID := filepath.Join(root, "cache"), testutil.ArtifactID(t, artifact.KindRecipe, "a picture objective's recipe").String()
	for _, run := range []struct {
		name, want string
		args       []string
	}{
		{"a proof under a recipe, without a cache", "is absent", []string{"-objective", pictureObjectiveAlias, "-proof", "-recipe", recipeID}},
		{"a proof given a cache", "a cache", []string{"-objective", pictureObjectiveAlias, "-proof", "-cache", cache}},
		{"a full run without a cache", "a cache", []string{"-objective", pictureObjectiveAlias}},
		{"a proof of a manifest's frames", "go with objective", []string{"-frames", filepath.Join(root, "frames.txt"), "-proof", "-cache", cache}},
		{"a recipe that is no identity", "training recipe", []string{"-objective", pictureObjectiveAlias, "-proof", "-recipe", "no identity"}},
	} {
		err := runRoute(t.Context(), append([]string{"codec", "-pair", filepath.Dir(pair.Bridge), "-batch", "2", "-steps", "1",
			"-store", filepath.Join(root, "store"), "-out", filepath.Join(root, "out")}, run.args...))
		if err == nil || !strings.Contains(err.Error(), run.want) {
			t.Errorf("%s = %v, want a refusal that says %q", run.name, err, run.want)
		}
	}
	if _, err := os.Stat(cache); !os.IsNotExist(err) {
		t.Errorf("a refused run left a token cache at %s (%v)", cache, err)
	}
}

// tinyTorchWeights is a torch state_dict holding one float32 weight of shape
// [1]: the smallest file an inventory reads as PyTorch weights.
func tinyTorchWeights(t *testing.T, value float32) []byte {
	t.Helper()
	var pickle bytes.Buffer
	pickle.Write([]byte{0x80, 2, '}', 'q', 0, '('}) // PROTO 2, an empty dict, MARK
	writePickleTensor(&pickle, "weight", "0", 1, []int64{1}, []int64{1})
	pickle.Write([]byte{'u', '.'}) // SETITEMS, STOP
	var archive bytes.Buffer
	files := zip.NewWriter(&archive)
	for _, entry := range []struct {
		name string
		data []byte
	}{{"archive/data.pkl", pickle.Bytes()}, {"archive/data/0", binary.LittleEndian.AppendUint32(nil, math.Float32bits(value))}} {
		writer, err := files.CreateHeader(&zip.FileHeader{Name: entry.name, Method: zip.Store})
		if err == nil {
			_, err = writer.Write(entry.data)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := files.Close(); err != nil {
		t.Fatal(err)
	}
	return archive.Bytes()
}

// writeBridgedPair lays out under models the files a bridged pair is: a
// collection with its decoder and latent statistics, the image processor of
// the encoder beside it, another encoder, and the pair's own directory with
// the bridge that joins them. Each file is a few bytes no loader would run;
// an identity reads them all the same.
func writeBridgedPair(t *testing.T, models string) rae.BridgedPair {
	t.Helper()
	write := func(path string, data []byte) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// The checkpoint a collection is known by and the encoder beside it, as
	// internal/rae/recipes/sampler.json declares them.
	collectionDirectory := filepath.Join(models, "collection")
	write(filepath.Join(collectionDirectory, "DiTs", "Dinov2", "wReg_base", "ImageNet256", "DiTDH-S_ep20", "stage2_model.pt"), []byte("a checkpoint only looked for"))
	write(filepath.Join(models, "dinov2-with-registers-base", "preprocessor_config.json"), []byte(`{"own":"processor"}`))
	collection, held, err := rae.ResolveCollection(collectionDirectory)
	if err != nil || !held {
		t.Fatalf("the collection did not resolve (%t, %v)", held, err)
	}
	write(collection.Decoder, tinyTorchWeights(t, 1))
	write(collection.Statistics, tinyTorchWeights(t, 2))
	encoder := filepath.Join(models, "other-encoder")
	write(filepath.Join(encoder, "config.json"), []byte(`{"source":"config"}`))
	write(filepath.Join(encoder, "preprocessor_config.json"), []byte(`{"source":"processor"}`))
	if err := safetensors.Save(filepath.Join(encoder, "model.safetensors"), map[string][]float32{"weight": {1}}, map[string][]int{"weight": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	pairDirectory := filepath.Join(models, "pair")
	if err := rae.SaveEncoderBridge(pairDirectory, rae.EncoderBridge{Weight: []float32{1, 0, 2, 0, -1, 0.5}, Bias: []float32{10, 20}}, "other-encoder", "collection"); err != nil {
		t.Fatal(err)
	}
	pair, held, err := rae.ResolveBridgedPair(pairDirectory)
	if err != nil || !held {
		t.Fatalf("the pair did not resolve (%t, %v)", held, err)
	}
	return pair
}

// TestCodecCheckpointNamesEveryFileOfThePairItTrained runs a picture
// objective's held-out run over a bridged pair the way the codec route does,
// by the pair's bridge file, and holds what it publishes to the whole pair: the
// checkpoint the verdict judges names the model the pair is, and once the
// pair's decoder is replaced by another of the same shape that checkpoint no
// longer names the pair. A checkpoint that named the bridge alone would go on
// standing for a pair whose decoder it never ran through.
func TestCodecCheckpointNamesEveryFileOfThePairItTrained(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const pictures, side = 12, 8
	objective, _ := registerPictureObjective(t, ctx, store, root, pictures, side)
	data, _, err := codecPictures(ctx, store, objective, side)
	if err != nil {
		t.Fatal(err)
	}
	pair := writeBridgedPair(t, filepath.Join(root, "models"))

	// The base model draws every value a tenth off and the trained one a
	// hundredth, so the run has a verdict to publish.
	off := 0.1
	var checkpoint trainingprogram.Checkpoint
	result, err := trainingworkflow.RunHeldout(ctx, store, trainingworkflow.HeldoutRun{
		Alias: pictureObjectiveAlias, ModelPath: pair.Bridge, Data: data, Host: true,
		Train: func(ctx context.Context, recipeID artifact.ID, shown []trainingworkflow.HeldoutExample) (trainingprogram.Checkpoint, error) {
			config, err := trainingprogram.BuiltinOptimizerPolicy().Config(testMatrix.End)
			if err != nil {
				return trainingprogram.Checkpoint{}, err
			}
			matrix := newHostMatrix(t, trainingprogram.ObjectiveImageLatent, config)
			step := matrix.step()
			off = 0.01
			checkpoint, err = publishPlannedTrained(ctx, store, recipeID, filepath.Join(root, "checkpoint"), matrix, trainingdata.StreamState{Identity: data.Training, Position: uint64(step.Step)}, false)
			return checkpoint, err
		},
		Predict: func(example trainingworkflow.HeldoutExample) (trainingworkflow.HeldoutPrediction, error) {
			drawn := make([]float64, len(example.Target))
			for index, value := range example.Target {
				drawn[index] = value + off
			}
			return trainingworkflow.HeldoutPrediction{Values: drawn}, nil
		},
	})
	if err != nil || !result.Passed || !result.Verdict.Valid() || result.Trained != checkpoint.ID() {
		t.Fatalf("held-out run over the pair = (passed=%t, verdict %s, trained %s, %v), want a passed verdict on checkpoint %s", result.Passed, result.Verdict, result.Trained, err, checkpoint.ID())
	}
	trained, err := trainingworkflow.WeightsIdentity(pair.Bridge)
	if err != nil || checkpoint.Model != trained {
		t.Fatalf("the checkpoint names model %s, and the pair it trained is %s (%v)", checkpoint.Model, trained, err)
	}
	if err := os.WriteFile(pair.Collection.Decoder, tinyTorchWeights(t, 3), 0o600); err != nil {
		t.Fatal(err)
	}
	replaced, err := trainingworkflow.WeightsIdentity(pair.Bridge)
	if err != nil || replaced == checkpoint.Model {
		t.Fatalf("with another decoder of the same shape the pair is model %s (%v), which the checkpoint trained before the replacement still names", replaced, err)
	}
}
