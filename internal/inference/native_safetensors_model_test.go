//go:build modeltest

package inference

import (
	"bytes"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modeltest"
	"overgo/internal/sampling"
)

// TestQwen35ServesFromItsSafetensors holds Qwen3.5-4B served from its own safetensors, a directory registered as
// itself and read through the converter's view, to the model its conversion served. Every tensor the view serves is the
// conversion's, byte for byte, but the draft layer's norms, which dcf1a1d66 shifted by one as the trunk's always were,
// so the trunk computes what the GGUF-served 4B computed, bit for bit, through the same runner. Opened without judging
// its drafting (OpenOptions.JudgeOwnDrafts, under which it decodes plainly), its own draft layer drafts by default and
// keeps at least one draft in ten, the floor below which it stops drafting, and its speculative decode emits what its
// plain decode does.
func TestQwen35ServesFromItsSafetensors(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	native := modeltest.Directory(t, "Qwen3.5-4B")
	converted := modeltest.Directory(t, "Qwen3.5-4B-f16.gguf")
	viewed, err := gguf.Open(native)
	if err != nil {
		t.Fatal(err)
	}
	defer viewed.Close()
	if viewed.ViewDirectory() == "" {
		t.Fatalf("%s is not served as the view of its own files", native)
	}
	written, err := gguf.Open(converted)
	if err != nil {
		t.Fatal(err)
	}
	defer written.Close()
	blocks, _ := viewed.MetadataValue("qwen35.block_count")
	drafts, _ := viewed.MetadataValue("qwen35.nextn_predict_layers")
	draftLayer := fmt.Sprintf("blk.%v.", blocks.Data.(uint32)-drafts.Data.(uint32))
	if len(viewed.Tensors) != len(written.Tensors) {
		t.Fatalf("the view serves %d tensors, the conversion %d", len(viewed.Tensors), len(written.Tensors))
	}
	var shifted, promoted []string
	for _, tensor := range written.Tensors {
		served, found := viewed.Tensor(tensor.Name)
		if !found || served.Dimensions != tensor.Dimensions || served.Shape != tensor.Shape {
			t.Fatalf("the view serves %s as %+v, the conversion as %+v", tensor.Name, served, tensor)
		}
		want, got := make([]byte, tensor.Size), make([]byte, served.Size)
		if err := written.ReadTensorData(tensor, want); err != nil {
			t.Fatal(err)
		}
		if err := viewed.ReadTensorData(served, got); err != nil {
			t.Fatal(err)
		}
		if served.Type == tensor.Type && bytes.Equal(want, got) {
			continue
		}
		// A tensor the converter now stores in another type, as it promotes a
		// recurrent layer's convolution to the F32 its kernel reads, holds the
		// same values when each decodes to the same floats.
		if served.Type != tensor.Type && sameValues(t, tensor, want, served, got) {
			promoted = append(promoted, tensor.Name)
			continue
		}
		if !strings.HasPrefix(tensor.Name, draftLayer) || !strings.Contains(tensor.Name, "norm") {
			t.Fatalf("the view's %s differs from the conversion's, outside the draft layer's norms", tensor.Name)
		}
		shifted = append(shifted, tensor.Name)
	}
	t.Logf("%d tensors byte for byte the conversion's, %d the same values in the type the converter now stores; the draft layer's norms the fix shifted: %v",
		len(written.Tensors)-len(shifted)-len(promoted), len(promoted), shifted)

	runner, err := openNativeFixtureRunner(native, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	if !runner.SpeculatesByDefault() {
		t.Fatal("the 4B served from its own files does not draft by default")
	}
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	drafted, kept := 0, 0
	byDefault := GenerateOptions{
		MaxNewTokens: 96, Sampler: greedy, DeviceGreedy: true,
		OnDraftSpan: func(spanDrafts, accepted int) { drafted, kept = drafted+spanDrafts, kept+accepted },
	}
	const prompt = "Explain, step by step, why the sum of the first n odd numbers is n squared."
	speculated, _, err := runner.Generate(t.Context(), prompt, byDefault)
	if err != nil {
		t.Fatal(err)
	}
	if drafted == 0 || kept*10 < drafted {
		t.Fatalf("the draft layer kept %d of %d drafts, want at least one in ten", kept, drafted)
	}
	plain := byDefault
	plain.PlainDecode, plain.OnDraftSpan = true, nil
	decoded, _, err := runner.Generate(t.Context(), prompt, plain)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(speculated, decoded) {
		t.Fatalf("the speculative decode emits %v, the plain one %v", speculated, decoded)
	}
	t.Logf("the draft layer kept %d of %d drafts over %d tokens", kept, drafted, len(speculated))
}

// sameValues reports whether two tensors' stored bytes decode to the same floats, bit for bit.
func sameValues(t *testing.T, left gguf.TensorInfo, leftBytes []byte, right gguf.TensorInfo, rightBytes []byte) bool {
	t.Helper()
	leftValues, err := model.DecodeHostTensor(left, leftBytes)
	if err != nil {
		t.Fatal(err)
	}
	rightValues, err := model.DecodeHostTensor(right, rightBytes)
	if err != nil {
		t.Fatal(err)
	}
	return slices.EqualFunc(leftValues.Data, rightValues.Data, func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) })
}
