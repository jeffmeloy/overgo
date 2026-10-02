package main

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/e4btrain"
	"overgo/internal/gguf"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/optimizer"
	"overgo/internal/quant"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// The e4b route publishes its trained model in its GGUF base's layout: its
// trainer names every tensor it trains as the artifact does, or the route
// would fall back to a trained-tensors-only weights file.
var _ servable = (*e4btrain.Trainer)(nil)

// The tiny Gemma-4 E4B the hermetic publication trains from: the real
// model's layer kinds in miniature. A sliding layer and a full layer
// project their own keys and values, and a sliding and a full layer after
// them attend those, as the real model's last 18 layers attend the last
// two that project. The full layers rotate under frequency factors that
// stop their trailing pairs, every layer carries a per-layer input adapter
// and all but the last scale their output.
const (
	tinyE4BHidden       = 16
	tinyE4BVocab        = 23
	tinyE4BPerLayer     = 8
	tinyE4BHeads        = 4
	tinyE4BKVHeads      = 2
	tinyE4BSlidingWidth = 8
	tinyE4BFullWidth    = 16
	tinyE4BInter        = 32
	tinyE4BWindow       = 3
	tinyE4BLayers       = 4
	tinyE4BSharedLayers = 2
	tinyE4BSoftcap      = 3
	tinyE4BStoppedPair  = 1e30
)

// writeTinyE4B writes the tiny Gemma-4 E4B at directory as the real model
// stores itself: bf16 tables, matrices and norms, f32 output scales and
// frequency factors, and every layer's key norm, which the layers that
// attend another layer's keys declare but never read.
func writeTinyE4B(t *testing.T, directory string) string {
	t.Helper()
	rng := rand.New(rand.NewSource(7))
	var tensors []gguf.TensorData
	add := func(name string, kind gguf.DType, deviation float64, shape ...uint64) {
		elements := uint64(1)
		for _, extent := range shape {
			elements *= extent
		}
		values := make([]float32, elements)
		for index := range values {
			values[index] = float32(rng.NormFloat64() * deviation)
			if len(shape) == 1 {
				values[index]++
			}
		}
		if name == "rope_freqs.weight" {
			for pair := range values {
				values[pair] = 1
				if pair >= len(values)/2 {
					values[pair] = tinyE4BStoppedPair
				}
			}
		}
		stored, err := quant.Quantize(kind, values)
		if err != nil {
			t.Fatal(err)
		}
		tensors = append(tensors, gguf.TensorData{Name: name, Shape: shape, Type: kind, Data: bytes.NewReader(stored)})
	}
	perLayerWidth := uint64(tinyE4BLayers * tinyE4BPerLayer)
	add("token_embd.weight", gguf.DTypeBF16, 0.5, tinyE4BHidden, tinyE4BVocab)
	add("per_layer_token_embd.weight", gguf.DTypeBF16, 0.5, perLayerWidth, tinyE4BVocab)
	add("per_layer_model_proj.weight", gguf.DTypeBF16, 0.25, tinyE4BHidden, perLayerWidth)
	add("per_layer_proj_norm.weight", gguf.DTypeBF16, 0.1, tinyE4BPerLayer)
	add("rope_freqs.weight", gguf.DTypeF32, 0, tinyE4BFullWidth/2)
	sliding := []bool{true, false, true, false}
	for layer := range tinyE4BLayers {
		name := func(role string) string { return fmt.Sprintf("blk.%d.%s.weight", layer, role) }
		width := uint64(tinyE4BFullWidth)
		if sliding[layer] {
			width = tinyE4BSlidingWidth
		}
		query, key := tinyE4BHeads*width, tinyE4BKVHeads*width
		add(name("attn_norm"), gguf.DTypeBF16, 0.1, tinyE4BHidden)
		add(name("attn_q"), gguf.DTypeBF16, 0.25, tinyE4BHidden, query)
		add(name("attn_q_norm"), gguf.DTypeBF16, 0.1, width)
		add(name("attn_k_norm"), gguf.DTypeBF16, 0.1, width)
		if layer < tinyE4BLayers-tinyE4BSharedLayers {
			add(name("attn_k"), gguf.DTypeBF16, 0.25, tinyE4BHidden, key)
			add(name("attn_v"), gguf.DTypeBF16, 0.25, tinyE4BHidden, key)
		}
		add(name("attn_output"), gguf.DTypeBF16, 0.25, query, tinyE4BHidden)
		add(name("post_attention_norm"), gguf.DTypeBF16, 0.1, tinyE4BHidden)
		add(name("ffn_norm"), gguf.DTypeBF16, 0.1, tinyE4BHidden)
		add(name("ffn_gate"), gguf.DTypeBF16, 0.25, tinyE4BHidden, tinyE4BInter)
		add(name("ffn_up"), gguf.DTypeBF16, 0.25, tinyE4BHidden, tinyE4BInter)
		add(name("ffn_down"), gguf.DTypeBF16, 0.25, tinyE4BInter, tinyE4BHidden)
		add(name("post_ffw_norm"), gguf.DTypeBF16, 0.1, tinyE4BHidden)
		add(name("per_layer_inp_gate"), gguf.DTypeBF16, 0.25, tinyE4BHidden, tinyE4BPerLayer)
		add(name("per_layer_proj"), gguf.DTypeBF16, 0.25, tinyE4BPerLayer, tinyE4BHidden)
		add(name("per_layer_post_norm"), gguf.DTypeBF16, 0.1, tinyE4BHidden)
		if layer < tinyE4BLayers-1 {
			add(name("layer_output_scale"), gguf.DTypeF32, 0.1, 1)
		}
	}
	add("output_norm.weight", gguf.DTypeBF16, 0.1, tinyE4BHidden)
	layerInts := func(value int32) []int32 {
		values := make([]int32, tinyE4BLayers)
		for index := range values {
			values[index] = value
		}
		return values
	}
	metadata := []gguf.Metadata{
		testutil.GGUFScalar("general.architecture", gguf.ValueTypeString, "gemma4"),
		testutil.GGUFScalar("gemma4.block_count", gguf.ValueTypeUint32, uint32(tinyE4BLayers)),
		testutil.GGUFScalar("gemma4.context_length", gguf.ValueTypeUint32, uint32(64)),
		testutil.GGUFScalar("gemma4.embedding_length", gguf.ValueTypeUint32, uint32(tinyE4BHidden)),
		testutil.GGUFArray("gemma4.feed_forward_length", gguf.ValueTypeInt32, layerInts(tinyE4BInter)),
		testutil.GGUFScalar("gemma4.attention.head_count", gguf.ValueTypeUint32, uint32(tinyE4BHeads)),
		testutil.GGUFArray("gemma4.attention.head_count_kv", gguf.ValueTypeInt32, layerInts(tinyE4BKVHeads)),
		testutil.GGUFScalar("gemma4.attention.key_length", gguf.ValueTypeUint32, uint32(tinyE4BFullWidth)),
		testutil.GGUFScalar("gemma4.attention.value_length", gguf.ValueTypeUint32, uint32(tinyE4BFullWidth)),
		testutil.GGUFScalar("gemma4.attention.key_length_swa", gguf.ValueTypeUint32, uint32(tinyE4BSlidingWidth)),
		testutil.GGUFScalar("gemma4.attention.value_length_swa", gguf.ValueTypeUint32, uint32(tinyE4BSlidingWidth)),
		testutil.GGUFScalar("gemma4.rope.dimension_count", gguf.ValueTypeUint32, uint32(tinyE4BFullWidth)),
		testutil.GGUFScalar("gemma4.rope.dimension_count_swa", gguf.ValueTypeUint32, uint32(tinyE4BSlidingWidth)),
		testutil.GGUFScalar("gemma4.rope.freq_base", gguf.ValueTypeFloat32, float32(1_000_000)),
		testutil.GGUFScalar("gemma4.rope.freq_base_swa", gguf.ValueTypeFloat32, float32(10_000)),
		testutil.GGUFScalar("gemma4.attention.sliding_window", gguf.ValueTypeUint32, uint32(tinyE4BWindow)),
		testutil.GGUFArray("gemma4.attention.sliding_window_pattern", gguf.ValueTypeBool, sliding),
		testutil.GGUFScalar("gemma4.attention.shared_kv_layers", gguf.ValueTypeUint32, uint32(tinyE4BSharedLayers)),
		testutil.GGUFScalar("gemma4.embedding_length_per_layer_input", gguf.ValueTypeUint32, uint32(tinyE4BPerLayer)),
		testutil.GGUFScalar("gemma4.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, float32(1e-6)),
		testutil.GGUFScalar("gemma4.final_logit_softcapping", gguf.ValueTypeFloat32, float32(tinyE4BSoftcap)),
		testutil.GGUFScalar("gemma4.vocab_size", gguf.ValueTypeUint32, uint32(tinyE4BVocab)),
	}
	path := filepath.Join(directory, "tiny-e4b.gguf")
	if err := gguf.WriteFileExclusive(path, metadata, tensors, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	return path
}

// registeredCandidate is the inference candidate the store registered for
// the model the checkpoint produced.
func registeredCandidate(t *testing.T, store artifact.Reader, checkpoint artifact.ID) recipe.Definition {
	t.Helper()
	ctx := t.Context()
	children, err := store.Children(ctx, checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	for _, produced := range children {
		if produced.Child.Kind() != artifact.KindModel || produced.Relation != artifact.RelationProducedBy {
			continue
		}
		dependents, err := store.Children(ctx, produced.Child)
		if err != nil {
			t.Fatal(err)
		}
		for _, dependent := range dependents {
			if dependent.Child.Kind() != artifact.KindRecipe {
				continue
			}
			definition, err := recipe.RequireDefinition(ctx, store, dependent.Child)
			if err != nil {
				t.Fatal(err)
			}
			if definition.Task == recipe.TaskInference && definition.Model == produced.Child {
				return definition
			}
		}
	}
	t.Fatalf("checkpoint %s produced no model registered as an inference candidate", checkpoint)
	return recipe.Definition{}
}

// unstepped is an e4b model published before any step: its weights are the
// ones it loaded and its momentum is zero.
type unstepped struct {
	*e4btrain.Model
	config optimizer.Config
}

// Snapshot is the optimizer state before the first step.
func (u unstepped) Snapshot() optimizer.State {
	return optimizer.State{PlanIdentity: u.OptimizerPlan().Identity(), Config: u.config, MomentumF32: make([]float32, u.ParameterCount())}
}

// TestE4BZeroStepPublishKeepsEveryTensorAsStored publishes the tiny E4B
// through the e4b route's session before any step, into a store holding
// the architecture profiles, and holds what it publishes to its base: the
// trained model opens as its two shards, every tensor keeps the base's
// stored type and bytes, so the loader read each as the base stores it
// and the publication rewrote none, and the second shard holds exactly the
// tensors the trainer never trains. The model registers as a candidate
// produced by the checkpoint, which once activated the server resolves,
// and a second publication links the first's second shard rather than
// writing it again.
func TestE4BZeroStepPublishKeepsEveryTensorAsStored(t *testing.T) {
	t.Parallel()
	root, store, _, datasetPath := routeFixture(t)
	base := writeTinyE4B(t, root)
	ctx := t.Context()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	session, err := openSession(ctx, store, base, datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	window := []uint32{1, 5, 9, 2, 22, 7, 3, 11, 4}
	model, err := e4btrain.LoadArtifact(ctx, base, window)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: session.authority.Objective, Updates: 1, MaximumSequence: len(window),
		Parameters: model.ParameterCount(), Optimizer: session.authority.Optimizer,
	})
	if err != nil {
		t.Fatal(err)
	}
	trainer := unstepped{Model: model, config: plan.Optimizer()}
	first, second := filepath.Join(root, "first"), filepath.Join(root, "second")
	for _, target := range []string{first, second} {
		if err := session.publishPlanned(ctx, target, trainer, plan.Updates()); err != nil {
			t.Fatal(err)
		}
	}
	var settled error
	if session.settle(ctx, &settled); settled != nil {
		t.Fatal(settled)
	}

	checkpoint, err := trainingprogram.LoadCheckpoint(first)
	if err != nil {
		t.Fatal(err)
	}
	original, err := gguf.Open(base)
	if err != nil {
		t.Fatal(err)
	}
	defer original.Close()
	published, err := gguf.Open(filepath.Join(first, trainingprogram.CheckpointModel))
	if err != nil {
		t.Fatal(err)
	}
	defer published.Close()
	trained := map[string]bool{}
	for _, group := range model.OptimizerPlan().Groups() {
		trained[group.Name] = true
	}
	var frozen []string
	for _, info := range original.Tensors {
		stored := make([]byte, info.Size)
		if err := original.ReadTensorData(info, stored); err != nil {
			t.Fatal(err)
		}
		held, ok := published.Tensor(info.Name)
		written := make([]byte, held.Size)
		if !ok || published.ReadTensorData(held, written) != nil || held.Type != info.Type || !bytes.Equal(written, stored) {
			t.Fatalf("%s publishes as %s, want the base's %s bytes unchanged", info.Name, held.Type, info.Type)
		}
		want := uint16(0)
		if !trained[info.Name] {
			want = 1
			frozen = append(frozen, info.Name)
		}
		if held.Shard != want {
			t.Fatalf("%s is in shard %d, want %d: the first holds what the trainer trains, the second what it never trains", info.Name, held.Shard, want)
		}
	}
	wantFrozen := []string{"token_embd.weight", "per_layer_token_embd.weight", "rope_freqs.weight", "blk.2.attn_k_norm.weight", "blk.3.attn_k_norm.weight"}
	if !slices.Equal(slices.Sorted(slices.Values(frozen)), slices.Sorted(slices.Values(wantFrozen))) {
		t.Fatalf("the trainer leaves %v untrained, want the lexical tables, the frequency factors and the shared layers' key norms %v", frozen, wantFrozen)
	}
	// The server resolves the registered candidate once it is active.
	candidate := registeredCandidate(t, store, checkpoint.ID())
	if err := modelrecipetest.ActivateCandidate(ctx, store, "fixture/tiny-e4b", candidate); err != nil {
		t.Fatal(err)
	}
	served, err := modelrecipe.ResolveActiveGGUF(ctx, store, filepath.Join(first, trainingprogram.CheckpointModel))
	if err != nil {
		t.Fatal(err)
	}
	if err := served.Close(); err != nil {
		t.Fatal(err)
	}
	leftInfo, leftErr := os.Stat(filepath.Join(first, trainingprogram.CheckpointFrozenShard))
	rightInfo, rightErr := os.Stat(filepath.Join(second, trainingprogram.CheckpointFrozenShard))
	if leftErr != nil || rightErr != nil || !os.SameFile(leftInfo, rightInfo) {
		t.Fatalf("the second publication's frozen shard = (%v, %v), want the first's linked", leftErr, rightErr)
	}
}
