package inference

import (
	"context"
	"os"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

func TestValidateStep35MTP(t *testing.T) {
	runner := &Runner{
		spec:    model.Spec{Architecture: "step35", NextNPredictLayers: 2},
		weights: model.Weights{Step35MTP: make([]model.Step35MTPWeights, 2)},
	}
	if err := runner.validateStep35MTP(); err != nil {
		t.Fatal(err)
	}
	runner.weights.Step35MTP = runner.weights.Step35MTP[:1]
	if err := runner.validateStep35MTP(); err == nil {
		t.Fatal("incomplete Step3.5 MTP head catalog was accepted")
	}
}

func TestLastValueColumn(t *testing.T) {
	value := reference.Value{
		Shape: tensor.MustShape(2, 3),
		Data:  []float32{1, 2, 3, 4, 5, 6},
	}
	last, err := lastValueColumn(value)
	if err != nil {
		t.Fatal(err)
	}
	if !last.Shape.Equal(tensor.MustShape(2, 1)) || len(last.Data) != 2 || last.Data[0] != 5 || last.Data[1] != 6 {
		t.Fatalf("unexpected final column: %+v", last)
	}
}

func TestStep35MTPChainsIndependentHeads(t *testing.T) {
	modelPath := os.Getenv("LLAMACPP2GO_STEP35_MTP_MODEL")
	if modelPath == "" {
		t.Skip("LLAMACPP2GO_STEP35_MTP_MODEL is not set")
	}
	runner, err := OpenWithOptions(modelPath, OpenOptions{
		DeviceOrdinal: 0, PreloadQuantizedWeights: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := context.Background()
	session, err := runner.NewStep35MTPSession(ctx, []tokenizer.TokenID{0})
	if err != nil {
		t.Fatal(err)
	}
	if len(session.Heads) != int(runner.spec.NextNPredictLayers) {
		t.Fatalf("Step3.5 MTP head count = %d", len(session.Heads))
	}
	logits, next, err := runner.AdvanceStep35MTP(ctx, 0, session)
	if err != nil {
		t.Fatal(err)
	}
	if logits.Shape.Dims[0] != uint64(runner.spec.VocabularySize) ||
		next.Position != session.Position+1 || len(next.DraftTokens) != 1 ||
		next.Heads[0].Key.Shape.Dims[2] != uint64(session.MTPStart+1) {
		t.Fatalf("unexpected Step3.5 head-0 state: logits=%v before=%+v after=%+v", logits.Shape, session, next)
	}
	if len(next.Heads) < 2 {
		return
	}
	_, third, err := runner.AdvanceStep35MTP(ctx, 0, next)
	if err != nil {
		t.Fatal(err)
	}
	if third.Position != next.Position+1 || len(third.DraftTokens) != 2 ||
		third.Heads[1].Key.Shape.Dims[2] != uint64(session.MTPStart+2) {
		t.Fatalf("Step3.5 head-1 cache did not rebuild the draft prefix: %+v", third)
	}
}
