package inference

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

func TestLoRAGraphAppliesAlphaScaleAndGlobalDisable(t *testing.T) {
	adapter := &model.LoRAAdapter{Path: "adapter.gguf", Alpha: 4, Weights: map[string]model.LoRAWeight{
		"projection": {
			A: reference.Value{Shape: tensor.MustShape(2, 2), Data: []float32{1, 0, 0, 1}},
			B: reference.Value{Shape: tensor.MustShape(2, 2), Data: []float32{2, 0, 0, 3}},
		},
	}}
	runner := &Runner{loraAdapters: []loadedLoRA{{adapter: adapter, scale: 0.5}}}
	builder := runner.newGraphBuilder()
	weight := builder.Input("projection", dtype.F32, tensor.MustShape(2, 2))
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	output := builder.MulMat(weight, input)
	results, err := reference.Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		weight: {Shape: weight.Shape, Data: []float32{1, 0, 0, 1}},
		input:  {Shape: input.Shape, Data: []float32{5, 7}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := results[output].Data, []float32{15, 28}; !reflect.DeepEqual(got, want) {
		t.Fatalf("adapted output = %v, want %v", got, want)
	}
	if err := runner.SetLoRAScales(nil); err != nil {
		t.Fatal(err)
	}
	if got := runner.LoRAAdapters()[0].Scale; got != 0 {
		t.Fatalf("disabled scale = %g", got)
	}
}

func TestLoRAHostEmbeddingAndLogitPaths(t *testing.T) {
	runner := &Runner{loraAdapters: []loadedLoRA{{adapter: &model.LoRAAdapter{
		Alpha: 2,
		Weights: map[string]model.LoRAWeight{
			"token_embd.weight": {
				A:         reference.Value{Shape: tensor.MustShape(1, 3), Data: []float32{2, 4, 6}},
				B:         reference.Value{Shape: tensor.MustShape(1, 2), Data: []float32{3, 5}},
				Embedding: true,
			},
			"output.weight": {
				A: reference.Value{Shape: tensor.MustShape(2, 1), Data: []float32{2, 3}},
				B: reference.Value{Shape: tensor.MustShape(1, 2), Data: []float32{4, 5}},
			},
		},
	}, scale: 0.25}}}
	embedded, err := runner.applyLoRAEmbeddingRows(
		"token_embd.weight", []uint32{1},
		reference.Value{Shape: tensor.MustShape(2, 1), Data: []float32{10, 20}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := embedded.Data, []float32{16, 30}; !reflect.DeepEqual(got, want) {
		t.Fatalf("embedding = %v, want %v", got, want)
	}
	logits := []float32{1, 2}
	if err := runner.applyLoRALogits("output.weight", []float32{2, 1}, logits); err != nil {
		t.Fatal(err)
	}
	if got, want := logits, []float32{15, 19.5}; !reflect.DeepEqual(got, want) {
		t.Fatalf("logits = %v, want %v", got, want)
	}
}

func TestSetLoRAScalesRejectsInvalidRequestsWithoutMutation(t *testing.T) {
	runner := &Runner{loraAdapters: []loadedLoRA{{adapter: &model.LoRAAdapter{Path: "a"}, scale: 1}}}
	for _, request := range [][]LoRAScale{
		{{ID: 1, Scale: 1}},
		{{ID: 0, Scale: 1}, {ID: 0, Scale: 2}},
	} {
		err := runner.SetLoRAScales(request)
		if err == nil || !strings.Contains(err.Error(), "LoRA adapter ID") {
			t.Fatalf("request %v error = %v", request, err)
		}
		if runner.loraAdapters[0].scale != 1 {
			t.Fatal("invalid request mutated scale")
		}
	}
}

func TestLoRAScaleBindsSessionSignature(t *testing.T) {
	adapter := &model.LoRAAdapter{Path: "adapter.gguf", Weights: map[string]model.LoRAWeight{
		"projection": {
			A: reference.Value{Shape: tensor.MustShape(1, 1), Data: []float32{2}},
			B: reference.Value{Shape: tensor.MustShape(1, 1), Data: []float32{3}},
		},
	}}
	runner := &Runner{spec: model.Spec{Architecture: "llama"}, loraAdapters: []loadedLoRA{{
		adapter: adapter, scale: 1, signature: loRAStaticSignature(adapter),
	}}}
	before, err := runner.sessionModelSignature()
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.SetLoRAScales([]LoRAScale{{ID: 0, Scale: 0.5}}); err != nil {
		t.Fatal(err)
	}
	after, err := runner.sessionModelSignature()
	if err != nil {
		t.Fatal(err)
	}
	if before == after {
		t.Fatal("LoRA scale did not change session signature")
	}
}

func TestGenerateRestoresPerRequestLoRA(t *testing.T) {
	runner := &Runner{
		spec:         model.Spec{Architecture: "llama", ContextLength: 8},
		vocab:        &tokenizer.Vocab{Tokens: []tokenizer.Token{{Text: "x", Type: tokenizer.TokenNormal}}},
		loraAdapters: []loadedLoRA{{adapter: &model.LoRAAdapter{Path: "adapter.gguf"}, scale: 1}},
	}
	_, _, err := runner.Generate(context.Background(), "", GenerateOptions{
		MaxNewTokens:   0,
		PromptTokenIDs: []tokenizer.TokenID{0},
		LoRA:           []LoRAScale{{ID: 0, Scale: 0.25}},
		LoRAConfigured: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if runner.loraAdapters[0].scale != 1 {
		t.Fatalf("global scale after request = %g", runner.loraAdapters[0].scale)
	}
}

func TestActiveALoRAUsesLastInvocationAndRejectsMultiple(t *testing.T) {
	adapter := func(path string, invocation ...uint32) loadedLoRA {
		return loadedLoRA{adapter: &model.LoRAAdapter{Path: path, InvocationTokens: invocation}, scale: 1}
	}
	runner := &Runner{loraAdapters: []loadedLoRA{adapter("a", 2, 3)}}
	id, start, err := runner.activeALoRA([]tokenizer.TokenID{1, 2, 3, 2, 3, 4})
	if err != nil || id != 0 || start != 3 {
		t.Fatalf("active aLoRA = id:%d start:%d err:%v", id, start, err)
	}
	_, start, err = runner.activeALoRA([]tokenizer.TokenID{1, 2, 4})
	if err != nil || start != -1 {
		t.Fatalf("missing invocation start/error = %d %v", start, err)
	}
	runner.loraAdapters = append(runner.loraAdapters, adapter("b", 5))
	if _, _, err := runner.activeALoRA([]tokenizer.TokenID{2, 3, 5}); err == nil {
		t.Fatal("multiple aLoRA adapters were accepted")
	}
	runner.loraAdapters[1].adapter.InvocationTokens = nil
	if id, start, err := runner.activeALoRA([]tokenizer.TokenID{2, 3}); err != nil || id != -1 || start != -1 {
		t.Fatalf("mixed ordinary/aLoRA activation = id:%d start:%d err:%v", id, start, err)
	}
}
