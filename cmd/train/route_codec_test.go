//go:build modeltest

package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/latentvideo"
	"overgo/internal/media"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/rae"
	"overgo/internal/safetensors"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestCodecRouteTrainsADecoderOrNamesTheMissingOperator answers, by running
// the route, which installed autoencoder the existing backward operators
// reach. The representation autoencoder's decoder has a backward pass, so
// the bridge that feeds it takes one Muon step on two real frames, the loss
// on them falls, and the step is published as a checkpoint the store
// records. The Wan autoencoder's decoder has none, and is refused by the name
// of its first operation.
func TestCodecRouteTrainsADecoderOrNamesTheMissingOperator(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	work := t.TempDir()
	required := []string{"codec", "-frames", filepath.Join(work, "frames.txt"), "-batch", "2", "-steps", "1",
		"-cache", filepath.Join(work, "cache"), "-store", filepath.Join(work, "store"), "-out", filepath.Join(work, "out")}
	if err := os.WriteFile(filepath.Join(work, "frames.txt"), []byte("first.mp4\nsecond.mp4\n"), 0o600); err != nil {
		t.Fatal(err)
	}
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
		pair := roots.ResolveModelPath("webssl-dino-to-rae-bridge")
		if _, held, err := rae.ResolveBridgedPair(pair); err != nil || !held {
			testskip.NotApplicable(t, "no bridged pair with its encoder and collection is at "+pair)
		}
		if _, err := media.ResolveFFmpeg(""); err != nil {
			testskip.NotApplicable(t, "no FFmpeg to cut training frames with")
		}
		clips, err := filepath.Glob(filepath.Join(roots.Datasets, "VidGen", "*.mp4"))
		if err != nil || len(clips) < 2 {
			testskip.NotApplicable(t, "no clips to train on under "+filepath.Join(roots.Datasets, "VidGen"))
		}
		testskip.Short(t, "loads the bridged encoder and the collection's 1.7 GB decoder and trains a step")
		for index, name := range []string{"first.mp4", "second.mp4"} {
			source, err := os.Open(clips[index])
			if err != nil {
				t.Fatal(err)
			}
			target, err := os.Create(filepath.Join(work, name))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := io.Copy(target, source); err != nil {
				t.Fatal(err)
			}
			if err := errorsJoinClose(source, target); err != nil {
				t.Fatal(err)
			}
		}
		if err := runRoute(t.Context(), append(required, "-pair", pair)); err != nil {
			t.Fatal(err)
		}
		out := filepath.Join(work, "out")
		checkpoint, err := trainingprogram.LoadCheckpoint(out)
		if err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.Open(filepath.Join(work, "store"))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if _, recorded, err := store.Artifact(t.Context(), checkpoint.ID()); err != nil || !recorded || checkpoint.Optimizer.Step != 1 {
			t.Fatalf("checkpoint %s at step %d recorded %t (%v), want the one-step bridge recorded", checkpoint.ID(), checkpoint.Optimizer.Step, recorded, err)
		}
		weights, err := safetensors.OpenFile(filepath.Join(out, trainingprogram.CheckpointWeights))
		if err != nil {
			t.Fatal(err)
		}
		defer weights.Close()
		if len(weights.Tensors) != 2 || checkpoint.ParameterCount != 768*1025 {
			t.Fatalf("the checkpoint holds %d tensors and %d parameters, want the bridge's weights and offsets", len(weights.Tensors), checkpoint.ParameterCount)
		}
	})
}

// errorsJoinClose closes both files and returns the first failure.
func errorsJoinClose(files ...*os.File) error {
	var first error
	for _, file := range files {
		if err := file.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
