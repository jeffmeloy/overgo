package scratchmodel

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"testing"
)

func TestScratchHostTrajectoryParity(t *testing.T) {
	oracle := loadOracle(t)
	facts := CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}
	construction, err := Compile(facts, AdaptiveDerivationProfile())
	if err != nil {
		t.Fatal(err)
	}
	first, err := construction.Probe(oracle.Train[:1])
	if err != nil {
		t.Fatal(err)
	}
	if !near(first.Loss, oracle.FirstLoss, 2e-15) {
		t.Fatalf("first loss = %.17g, want %.17g", first.Loss, oracle.FirstLoss)
	}
	if len(first.Logits) != 1 || len(first.Logits[0]) != construction.config.BlockSize {
		t.Fatalf("logit geometry = %d documents / %d positions", len(first.Logits), len(first.Logits[0]))
	}
	replay, err := construction.Probe(oracle.Train[:1])
	if err != nil || !reflect.DeepEqual(first.Logits, replay.Logits) {
		t.Fatal("logit replay differs")
	}

	probe, err := construction.Probe(oracle.Train)
	if err != nil {
		t.Fatal(err)
	}
	updates, err := construction.MuonProbe(oracle.Train)
	if err != nil {
		t.Fatal(err)
	}
	if len(probe.Groups) != len(oracle.Groups) || len(updates) != len(oracle.Groups) {
		t.Fatalf("probe groups = %d/%d, want %d", len(probe.Groups), len(updates), len(oracle.Groups))
	}
	for index, group := range probe.Groups {
		want := oracle.Groups[index]
		if group.Name != want.Name || quantizedDigest(group.Gradients, 1e12) != want.Gradient1e12SHA256 {
			t.Fatalf("gradient parity differs for %q", group.Name)
		}
		if updates[index].Name != want.Name || quantizedDigest(updates[index].Gradients, 1e8) != want.Update1e8SHA256 {
			t.Fatalf("first Muon update parity differs for %q", group.Name)
		}
	}

	continuous, err := construction.TrainHost(oracle.Steps, oracle.Steps, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertTrajectory(t, continuous.Losses, oracle.LossHistory)
	if !near(continuous.ValidationLoss, oracle.FinalValLoss, 2e-15) {
		t.Fatalf("validation loss = %.17g, want %.17g", continuous.ValidationLoss, oracle.FinalValLoss)
	}
	firstStep, err := construction.TrainHost(oracle.Steps, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := construction.TrainHost(oracle.Steps, oracle.Steps-1, &firstStep.Checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resumed.Losses, continuous.Losses[1:]) || resumed.ValidationLoss != continuous.ValidationLoss ||
		!reflect.DeepEqual(resumed.Checkpoint.Model.Weights, continuous.Checkpoint.Model.Weights) ||
		!reflect.DeepEqual(resumed.Checkpoint.Momentum, continuous.Checkpoint.Momentum) {
		t.Fatal("checkpoint resume differs from uninterrupted run")
	}

	invalid := firstStep.Checkpoint
	invalid.ProgramID = "recipe:sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
	if _, err := construction.TrainHost(oracle.Steps, oracle.Steps-1, &invalid); err == nil {
		t.Fatal("foreign checkpoint authority accepted")
	}
	invalid = firstStep.Checkpoint
	invalid.Momentum = cloneWeights(invalid.Momentum)
	invalid.Momentum[construction.Parameters()[0].Name][0] = math.NaN()
	if _, err := construction.TrainHost(oracle.Steps, oracle.Steps-1, &invalid); err == nil {
		t.Fatal("non-finite checkpoint momentum accepted")
	}
	invalid = firstStep.Checkpoint
	invalid.NSStage1 = cloneIntMap(invalid.NSStage1)
	invalid.NSStage1[construction.Parameters()[0].Name] = 13
	if _, err := construction.TrainHost(oracle.Steps, oracle.Steps-1, &invalid); err == nil {
		t.Fatal("invalid checkpoint Muon stage accepted")
	}
	secondConstruction, err := Compile(facts, AdaptiveDerivationProfile())
	if err != nil {
		t.Fatal(err)
	}
	secondRun, err := secondConstruction.TrainHost(oracle.Steps, oracle.Steps, nil)
	if err != nil || !reflect.DeepEqual(secondRun.Losses, continuous.Losses) ||
		!reflect.DeepEqual(secondRun.Checkpoint.Model.Weights, continuous.Checkpoint.Model.Weights) {
		t.Fatal("RNG replay differs")
	}
}

func quantizedDigest(values []float64, scale float64) string {
	digest := sha256.New()
	var encoded [8]byte
	for _, value := range values {
		binary.LittleEndian.PutUint64(encoded[:], uint64(int64(math.Round(value*scale))))
		digest.Write(encoded[:])
	}
	return fmt.Sprintf("%x", digest.Sum(nil))
}

func assertTrajectory(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("trajectory length = %d, want %d", len(got), len(want))
	}
	for index := range got {
		if !near(got[index], want[index], 2e-15) {
			t.Fatalf("loss[%d] = %.17g, want %.17g", index, got[index], want[index])
		}
	}
}

func near(left, right, tolerance float64) bool {
	return !math.IsNaN(left) && !math.IsInf(left, 0) && math.Abs(left-right) <= tolerance
}
