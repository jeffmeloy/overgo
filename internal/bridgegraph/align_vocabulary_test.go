package bridgegraph

import (
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/representation"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestAlignVocabularyProjection(t *testing.T) {
	sourceContent, source := bridgeContract(
		t, "align-source", representation.ModalityAudio, 2, 1, 3,
		representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative},
	)
	targetContent, target := bridgeContract(
		t, "align-target", representation.ModalityText, 3, 1, 3,
		representation.NormalizationContract{Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative},
	)
	vocabulary := bridgeTestID(t, artifact.KindTokenizer, "align-vocabulary")
	target.Authorities = []representation.Authority{{
		Role: representation.AuthorityTokenizer, Artifact: vocabulary,
	}}
	encodedTarget, err := json.Marshal(target)
	if err != nil {
		t.Fatal(err)
	}
	target, err = representation.ParseContract(encodedTarget)
	if err != nil {
		t.Fatal(err)
	}
	targetContent = encodedTarget
	head := bridgeTestID(t, artifact.KindTensorSet, "align-head")
	embedding := bridgeTestID(t, artifact.KindTensorSet, "target-embedding")
	definition := Definition{
		Source: source.ID, Target: target.ID, Operator: OperatorAlignVocabulary,
		Vocabulary: vocabulary, VocabularyHead: head, TargetEmbedding: embedding,
		VocabularySize: 4, VocabularyLimit: 4, EmbeddingMode: EmbeddingUntied, Bias: true,
	}
	program, err := (Compiler{}).Compile(definition, sourceContent, targetContent)
	if err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	input := builder.Input("source", dtype.F32, tensor.MustShape(2, 2))
	vocabularyHead := builder.Input("vocabulary-head", dtype.F32, tensor.MustShape(2, 4))
	vocabularyBias := builder.Input("vocabulary-bias", dtype.F32, tensor.MustShape(4))
	targetEmbedding := builder.Input("target-embedding", dtype.F32, tensor.MustShape(4, 3))
	output, err := program.Build(builder, input, Weights{
		First: vocabularyHead, FirstBias: vocabularyBias, Embedding: targetEmbedding,
	})
	if err != nil {
		t.Fatal(err)
	}
	ops := make([]tensor.Op, 0, len(builder.Nodes()))
	for _, node := range builder.Nodes() {
		ops = append(ops, node.Op)
	}
	if !slices.Contains(ops, tensor.OpSoftmax) {
		t.Fatalf("vocabulary graph operations=%v", ops)
	}
	values, err := reference.Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		input:          inferenceValue(t, input.Shape, []float32{1, 2, 3, 4}),
		vocabularyHead: reference.ZeroValue(vocabularyHead.Shape),
		vocabularyBias: reference.ZeroValue(vocabularyBias.Shape),
		targetEmbedding: inferenceValue(t, targetEmbedding.Shape, []float32{
			1, 1, 1, 1,
			2, 2, 2, 2,
			3, 3, 3, 3,
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values[output].Data, []float32{1, 2, 3, 1, 2, 3}) {
		t.Fatalf("embedding-basis output=%v", values[output].Data)
	}

	tooLarge := definition
	tooLarge.VocabularyLimit = tooLarge.VocabularySize - 1
	if _, err := (Compiler{}).Compile(tooLarge, sourceContent, targetContent); err == nil {
		t.Fatal("vocabulary above declared bound accepted")
	}
	wrongVocabulary := definition
	wrongVocabulary.Vocabulary = bridgeTestID(t, artifact.KindTokenizer, "wrong-vocabulary")
	if _, err := (Compiler{}).Compile(wrongVocabulary, sourceContent, targetContent); err == nil {
		t.Fatal("wrong target vocabulary accepted")
	}
	tied := definition
	tied.EmbeddingMode = EmbeddingTied
	if _, err := (Compiler{}).Compile(tied, sourceContent, targetContent); err == nil {
		t.Fatal("mismatched tied embedding tensors accepted")
	}
}

func inferenceValue(t *testing.T, shape tensor.Shape, data []float32) reference.Value {
	t.Helper()
	value, err := reference.NewValue(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	return value
}
