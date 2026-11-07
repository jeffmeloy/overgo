//go:build windows && modeltest

package hybridtrain

import (
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/inference"
	"overgo/internal/model"
	"overgo/internal/modelcli"
	"overgo/internal/modeltest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheHybridReferenceIsTheServedNineB holds the host reference, loaded from Qwen3.5-9B's file with every attention
// layer's output gate, to the runner serving that file on the card. A prompt opens with two placeholders whose
// embeddings are replaced by other tokens' rows, as a composite's bridge replaces them. At the last position the
// reference's normed hidden state stands within a small share of the served one's length and points its way, the
// logits both give over ids spread through the vocabulary lead with the same id, and replacing the placed rows with
// others moves both states alike.
func TestTheHybridReferenceIsTheServedNineB(t *testing.T) {
	// Serial: holds the 9B on the card and its F32 layers on the host.
	cudatest.Require(t)
	const placeholder, probes, distance, alignment, moved = "<|image_pad|>", 512, 5e-2, 0.998, 0.1
	path := modeltest.Directory(t, "Qwen3.5-9B-Q8_0.gguf")
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ctx := t.Context()
	runner := testcheck.Must(modelcli.OpenRunner(ctx, roots.Store, path, inference.OpenOptions{}))(t)
	defer runner.Close()
	prompt := placeholder + placeholder + "Describe this image in detail. The video shows a woman walking a dog along a beach at sunset, the waves rolling in behind them while gulls circle overhead and"
	ids := testcheck.Must(runner.TokenizeText(prompt, false, true))(t)
	tokens := make([]uint32, len(ids))
	for index, id := range ids {
		tokens[index] = uint32(id)
	}
	file := testcheck.Must(gguf.Open(path))(t)
	defer file.Close()
	spec := testcheck.Must(model.ReadSpec(file))(t)
	weights := testcheck.Must(model.ReadWeights(file, spec))(t)
	hidden := int(spec.EmbeddingLength)
	placedIDs := [][]uint32{{9729, 5440}, {11336, 13044}}
	placedRows := make([][]float32, len(placedIDs))
	served := make([][]float32, len(placedIDs))
	for index, pair := range placedIDs {
		rows := testcheck.Must(model.LoadHostRows(ctx, file, weights.TokenEmbedding, pair))(t).Data
		placedRows[index] = rows
		inputs := inference.ProjectedInputs{EmbeddingOverrides: []inference.EmbeddingOverride{
			{TokenIndex: 0, Embedding: rows[:hidden]}, {TokenIndex: 1, Embedding: rows[hidden:]},
		}}
		states, _, err := runner.ForwardCachedWithProjectedInputs(ctx, ids, nil, inputs)
		testcheck.NoError(t, err)
		served[index] = slices.Clone(states.Data[len(states.Data)-hidden:])
	}
	probed := make([]uint32, probes)
	for index := range probed {
		probed[index] = uint32(index * int(spec.VocabularySize) / probes)
	}
	head := testcheck.Must(model.LoadHostRows(ctx, file, *weights.Output, probed))(t).Data
	runner.Close()

	stack := testcheck.Must(LoadStackArtifact(ctx, path, tokens))(t)
	for layer, kind := range stack.Cfg.Types {
		if kind != FullAttention {
			continue
		}
		fused := testcheck.Must(model.LoadHostTensor(ctx, file, *weights.Layers[layer].AttentionQ))(t).Data
		headDim := stack.Cfg.HeadDim
		gate := make([]float32, stack.Cfg.Heads*headDim*hidden)
		for index := range stack.Cfg.Heads {
			copy(gate[index*headDim*hidden:(index+1)*headDim*hidden], fused[(index*2*headDim+headDim)*hidden:(index*2*headDim+2*headDim)*hidden])
		}
		stack.Weights[layer].Attn.Gate = gate
	}
	// The reference's vocabulary is the prompt's own tokens, each its own row.
	local := make([]int, len(tokens))
	for index := range local {
		local[index] = index
	}
	reference := &languageModel{
		Geometry: stack.Cfg, Vocab: len(tokens), Layers: stack.Weights,
		Embedding: testcheck.Must(model.LoadHostRows(ctx, file, weights.TokenEmbedding, tokens))(t).Data,
		FinalNorm: testcheck.Must(model.LoadHostTensor(ctx, file, weights.OutputNorm))(t).Data,
	}
	states := make([][]float32, len(placedRows))
	for index, rows := range placedRows {
		pass := testcheck.Must(reference.forward(local, []int{0, 1}, rows))(t)
		states[index] = make([]float32, hidden)
		hostmath.RMSNormInto(states[index], pass.out[len(pass.out)-hidden:], reference.FinalNorm, 1, hidden, stack.Cfg.Eps)
	}
	norm := func(values []float32) float64 {
		var sum float64
		for _, value := range values {
			sum += float64(value) * float64(value)
		}
		return math.Sqrt(sum)
	}
	gap := func(left, right []float32) []float32 {
		difference := make([]float32, len(left))
		for index := range left {
			difference[index] = left[index] - right[index]
		}
		return difference
	}
	for index := range placedRows {
		var dot float64
		for at := range hidden {
			dot += float64(states[index][at]) * float64(served[index][at])
		}
		apart, cosine := norm(gap(states[index], served[index]))/norm(served[index]), dot/(norm(states[index])*norm(served[index]))
		want, got := hostmath.Linear2(served[index], head, 1, hidden, probes), hostmath.Linear2(states[index], head, 1, hidden, probes)
		lead, ownLead := slices.Index(want, slices.Max(want)), slices.Index(got, slices.Max(got))
		t.Logf("placed rows %d: the reference's state stands %.4f of the served state's length from it, cosine %.5f, leading ids %d and %d", index, apart, cosine, probed[ownLead], probed[lead])
		if apart > distance || cosine < alignment || lead != ownLead {
			t.Errorf("placed rows %d: the reference's state stands %.4f from the served one's, at cosine %.5f, and leads with id %d where it leads with %d",
				index, apart, cosine, probed[ownLead], probed[lead])
		}
	}
	servedMove, referenceMove := norm(gap(served[1], served[0])), norm(gap(states[1], states[0]))
	if servedMove == 0 || math.Abs(referenceMove-servedMove) > moved*servedMove {
		t.Errorf("other placed rows move the served state by %.4g and the reference's by %.4g; want them moved alike", servedMove, referenceMove)
	}
}
