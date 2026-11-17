//go:build windows

package latentimage

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestAdaptersStackAsTheWorkflowsLoaderStacksThem holds stacked adapters (moss-ultra-compare/stacked-adapters), as the
// MOSS Ultra workflow's Turbo sections stack DMD2, moss-control and the SFT adapter: a denoiser given two adapters that
// name the same projection's tensors alike binds each one's tensors apart, the second's at its place, each scaled by a
// strength input of its own, and reads a weight the second replaces from the second's file; a tensor at its place is
// read from its file by its own name; two adapters replacing one weight are refused; and a request names its adapters
// in order, each at its strength (1 where it states none), or one adapter, not both.
func TestAdaptersStackAsTheWorkflowsLoaderStacksThem(t *testing.T) {
	spec := denoiserSpec()
	width := uint64(spec.Hidden)
	projection := func() map[string]lowRankPatch {
		return map[string]lowRankPatch{"layers.0.proj.q": {down: "q.lora_down.weight", up: "q.lora_up.weight", rank: 2, rows: width, in: width, out: width, scale: 1}}
	}
	first := &mediaAdapter{lowRank: projection(), diffs: map[string]string{}, replaced: map[string]replacement{}}
	second := &mediaAdapter{lowRank: projection(), diffs: map[string]string{}, replaced: map[string]replacement{
		"video_embedder.weight": {tensor: "video_embedder.weight", shape: []uint64{width, uint64(3 * spec.InChannels)}},
		"video_embedder.bias":   {tensor: "video_embedder.bias", shape: []uint64{width}},
	}}
	p := testcheck.Must(compileRotaryDenoiserProgram(spec, 1e-6, 2, 3, 4, nil, dtype.BF16, rotaryOptions{adapters: []*mediaAdapter{first, second}}))(t)
	if len(p.adapterInputs) != 2 || len(p.strengths) != 2 || p.strengths[0].Name != adapterStrengthInput || p.strengths[1].Name != adapterStrengthInput+"@1" {
		t.Fatalf("two adapters gave %d binding sets and the strengths %v; want two each, the second at its place", len(p.adapterInputs), p.strengths)
	}
	if p.adapterInputs[0].Node("q.lora_down.weight") == nil || p.adapterInputs[1].Node("q.lora_down.weight@1") == nil ||
		p.adapterInputs[0].Node("q.lora_down.weight@1") != nil || p.adapterInputs[1].Node("q.lora_down.weight") != nil {
		t.Error("the two adapters' tensors for one projection are not bound apart, each in its own set and the second at its place")
	}
	if replaced := p.adapterInputs[1].Node("video_embedder.weight@1"); replaced == nil || replaced.Type != dtype.BF16 || p.control == nil {
		t.Errorf("the weight the second adapter replaces is bound as %v with the control %v; want it from the second's file in the model's storage", replaced, p.control)
	}
	if twice := replacedTwice([]*mediaAdapter{second, second}); twice != "video_embedder.bias" && twice != "video_embedder.weight" {
		t.Errorf("two adapters replacing one weight were found as %q", twice)
	}
	if replacedTwice([]*mediaAdapter{first, second}) != "" {
		t.Error("adapters replacing different weights were refused")
	}

	path := writeAdapter(t, "placed", map[string][]float32{"q.lora_down.weight": {1, 2, 3, 4}}, map[string][]int{"q.lora_down.weight": {2, 2}})
	source := testcheck.Must(safetensors.OpenFile(path))(t)
	defer source.Close()
	b := tensor.NewBuilder()
	placed := b.Input("q.lora_down.weight@1", dtype.F32, tensor.MustShape(2, 2))
	payload := testcheck.Must(testcheck.Must(adapterPayload(source, placed))(t)())(t)
	want := make([]byte, 16)
	for index, value := range []float32{1, 2, 3, 4} {
		binary.LittleEndian.PutUint32(want[4*index:], math.Float32bits(value))
	}
	if !bytes.Equal(payload, want) {
		t.Errorf("a tensor at its place reads %v from its file; want the file's own %v", payload, want)
	}

	half := float32(0.5)
	stacked := Request{Prompt: "a fox", Width: 1024, Height: 576, Adapters: &[]AppliedAdapter{{Name: "moss-control"}, {Name: "8-steps-dmd2-lora", Strength: &half}}}
	testcheck.NoError(t, ValidateRequest(stacked))
	if applied := stacked.applied(); len(applied) != 2 || applied[0].Name != "moss-control" || applied[0].strength() != 1 || applied[1].strength() != half {
		t.Errorf("the request applies %+v; want moss-control at 1, then DMD2 at 0.5", applied)
	}
	if one := (Request{Adapter: "moss-control"}).applied(); len(one) != 1 || one[0].Name != "moss-control" || one[0].strength() != 1 {
		t.Errorf("one adapter applies as %+v", one)
	}
	both := stacked
	both.Adapter = "moss-control"
	for name, refused := range map[string]Request{"both forms": both, "an empty list": {Prompt: "a fox", Width: 1024, Height: 576, Adapters: &[]AppliedAdapter{}},
		"an unnamed adapter": {Prompt: "a fox", Width: 1024, Height: 576, Adapters: &[]AppliedAdapter{{}}}} {
		if ValidateRequest(refused) == nil {
			t.Errorf("a request with %s was taken", name)
		}
	}
	if !(Request{Adapters: &[]AppliedAdapter{{Name: "moss-control"}}}).rotaryOnly() {
		t.Error("stacked adapters were offered to a flow profile")
	}
}
