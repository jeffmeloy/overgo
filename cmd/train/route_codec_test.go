//go:build modeltest

package main

import (
	"fmt"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/rae"
	"overgo/internal/recipecontract"
	"overgo/internal/safetensors"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// installedBridgedPair is the installed bridged pair's directory and count
// clips copied beside a manifest that names them, or a skip that says which
// of the pair, FFmpeg and the clips this machine lacks.
func installedBridgedPair(t *testing.T, work string, count int) (pair, manifest string) {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	pair = roots.ResolveModelPath("webssl-dino-to-rae-bridge")
	if _, held, err := rae.ResolveBridgedPair(pair); err != nil || !held {
		testskip.NotApplicable(t, "no bridged pair with its encoder and collection is at "+pair)
	}
	if _, err := media.ResolveFFmpeg(""); err != nil {
		testskip.NotApplicable(t, "no FFmpeg to cut training frames with")
	}
	clips, err := filepath.Glob(filepath.Join(roots.Datasets, "VidGen", "*.mp4"))
	if err != nil || len(clips) < count {
		testskip.NotApplicable(t, "no clips to train on under "+filepath.Join(roots.Datasets, "VidGen"))
	}
	testskip.Short(t, "loads the bridged encoder and the collection's 1.7 GB decoder and trains a step")
	var names strings.Builder
	for index := range count {
		name := fmt.Sprintf("clip-%02d.mp4", index)
		source, err := os.Open(clips[index])
		if err != nil {
			t.Fatal(err)
		}
		target, err := os.Create(filepath.Join(work, name))
		if err != nil {
			t.Fatal(err)
		}
		_, copyErr := io.Copy(target, source)
		if err := firstError(copyErr, source.Close(), target.Close()); err != nil {
			t.Fatal(err)
		}
		names.WriteString(name + "\n")
	}
	manifest = filepath.Join(work, "frames.txt")
	if err := os.WriteFile(manifest, []byte(names.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	return pair, manifest
}

// firstError is the first failure among errs.
func firstError(errs ...error) error {
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// recordedCheckpoint loads the checkpoint a run published at out and holds
// the store to recording it.
func recordedBridgeCheckpoint(t *testing.T, storePath, out string) trainingprogram.Checkpoint {
	t.Helper()
	checkpoint, err := trainingprogram.LoadCheckpoint(out)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, recorded, err := store.Artifact(t.Context(), checkpoint.ID()); err != nil || !recorded {
		t.Fatalf("checkpoint %s recorded %t (%v), want the trained bridge recorded", checkpoint.ID(), recorded, err)
	}
	weights, err := safetensors.OpenFile(filepath.Join(out, trainingprogram.CheckpointWeights))
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	if len(weights.Tensors) != 2 || checkpoint.ParameterCount != 768*1025 {
		t.Fatalf("the checkpoint holds %d tensors and %d parameters, want the bridge's weights and offsets", len(weights.Tensors), checkpoint.ParameterCount)
	}
	return checkpoint
}

// TestCodecRouteTrainsADecoderOrNamesTheMissingOperator answers, by running
// the route, which installed autoencoder the existing backward operators
// reach. The representation autoencoder's decoder has a backward pass, so
// the bridge that feeds it takes one Muon step on two real frames, the loss
// on them falls, and the step is published as a checkpoint the store
// records. The Wan autoencoder's decoder has none, and is refused by the name
// of its first operation.
func TestCodecRouteTrainsADecoderOrNamesTheMissingOperator(t *testing.T) {
	work := t.TempDir()
	required := []string{"codec", "-frames", filepath.Join(work, "frames.txt"), "-batch", "2", "-steps", "1",
		"-cache", filepath.Join(work, "cache"), "-store", filepath.Join(work, "store"), "-out", filepath.Join(work, "out")}
	t.Run("the Wan autoencoder is refused by its first operation", func(t *testing.T) {
		wan := modeltest.Directory(t, "Wan2.1-T2V-1.3B")
		operations, err := latentvideo.AutoencoderDecoderOperations(wan)
		if err != nil || len(operations) == 0 {
			t.Fatalf("the Wan autoencoder's decoder names %d operations: %v", len(operations), err)
		}
		err = runRoute(t.Context(), append(required, "-pair", wan))
		if err == nil || !strings.Contains(err.Error(), "first operation, "+operations[0]+",") || !strings.Contains(err.Error(), "backward pass") {
			t.Fatalf("the Wan autoencoder was not refused by its first operation, %s: %v", operations[0], err)
		}
		t.Log(err)
	})
	t.Run("the bridge of the representation autoencoder takes a step", func(t *testing.T) {
		pair, _ := installedBridgedPair(t, work, 2)
		if err := runRoute(t.Context(), append(required, "-pair", pair)); err != nil {
			t.Fatal(err)
		}
		if checkpoint := recordedBridgeCheckpoint(t, filepath.Join(work, "store"), filepath.Join(work, "out")); checkpoint.Optimizer.Step != 1 {
			t.Fatalf("the checkpoint is at step %d, want the one step taken", checkpoint.Optimizer.Step)
		}
	})
}

// TestCodecRouteJudgesATrainedBridgeOnHeldOutFrames runs the codec route on
// the installed bridged pair from clips to a verdict. It cuts six clips'
// first frames into pictures of the pair's size, which register as a dataset
// split under a picture objective; the route then takes one Muon step on the
// training pictures and publishes a held-out verdict that scores the
// installed bridge and the trained one, on pictures neither was trained on
// here, by the peak signal-to-noise ratio of what the frozen decoder draws.
func TestCodecRouteJudgesATrainedBridgeOnHeldOutFrames(t *testing.T) {
	work := t.TempDir()
	const frames = 6
	pair, manifest := installedBridgedPair(t, work, frames)
	ctx := t.Context()
	pictures := filepath.Join(work, "pictures")
	if err := runRoute(ctx, []string{"codec", "-pair", pair, "-frames", manifest, "-cut", pictures}); err != nil {
		t.Fatal(err)
	}
	cut, err := filepath.Glob(filepath.Join(pictures, "*"+codecPictureExtension))
	if err != nil || len(cut) != frames {
		t.Fatalf("cut %d pictures (%v), want one for each of the %d clips", len(cut), err, frames)
	}
	for _, name := range cut {
		file, err := os.Open(name)
		if err != nil {
			t.Fatal(err)
		}
		shape, decodeErr := png.DecodeConfig(file)
		if err := firstError(decodeErr, file.Close()); err != nil || shape.Width != shape.Height || shape.Width == 0 {
			t.Fatalf("%s is %d by %d (%v), want a square picture", name, shape.Width, shape.Height, err)
		}
	}
	storePath, out := filepath.Join(work, "store"), filepath.Join(work, "out")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	source, training := registerSplit(t, ctx, store, "frames", pictures, frames, nil)
	registerObjective(t, ctx, store, work, pictureObjectiveAlias, trainingprogram.ObjectiveSpec{
		Name: "frames", Kind: trainingprogram.ObjectiveImageLatent,
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityImage}, Outputs: []recipecontract.Modality{recipecontract.ModalityImage}},
		Dataset:   source, Split: training, Metric: trainingprogram.MetricImagePSNRUnit,
	})
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := runRoute(ctx, []string{"codec", "-pair", pair, "-objective", pictureObjectiveAlias, "-batch", "2", "-steps", "1",
		"-cache", filepath.Join(work, "cache"), "-store", storePath, "-out", out}); err != nil {
		t.Fatal(err)
	}
	checkpoint := recordedBridgeCheckpoint(t, storePath, out)
	if store, err = overgodb.Open(storePath); err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	verdicts := 0
	metric := string(trainingprogram.MetricImagePSNRUnit)
	err = evaluation.VisitHeldoutVerdicts(ctx, store, func(_ artifact.ID, base, trained evaluation.EvalSummary) error {
		verdicts++
		installed, adapted := base.Metrics[metric], trained.Metrics[metric]
		t.Logf("held-out pictures: the installed bridge %.3f dB, the bridge after one step %.3f dB", installed, adapted)
		// The installed pair draws real frames between a flat picture's ratio and a pixel codec's.
		if math.IsNaN(installed) || installed < 10 || installed > 30 || math.IsNaN(adapted) || adapted == installed {
			t.Errorf("the verdict reads %v dB for the installed bridge and %v dB for the trained one", installed, adapted)
		}
		return nil
	}, nil)
	if err != nil || verdicts != 1 || checkpoint.Optimizer.Step != 1 {
		t.Fatalf("read %d verdicts (%v) of a checkpoint at step %d, want one verdict of the one step", verdicts, err, checkpoint.Optimizer.Step)
	}
}
