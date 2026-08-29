package modelmerge

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/evaluation"
	"overgo/internal/model"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

func TestDenseToMoEEvidenceScreen(t *testing.T) {
	candidate := denseToMoECandidateFixture(t)
	if err := candidate.ValidateGeneratedArtifact(); err != nil {
		t.Fatal(err)
	}
	for _, removed := range []string{denseGateName, denseUpName, denseDownName} {
		if _, found := candidate.Model.Tensors[removed]; found {
			t.Fatalf("dense tensor %q remains in generated model", removed)
		}
	}
	gate := candidate.Model.Tensors[expertGateName]
	if !tensor.HasDimensions(gate.Layout, denseEmbeddingWidth, denseFeedForwardWidth, denseExpertCount) ||
		!slices.Equal(gate.Values, []float32{1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16}) {
		t.Fatalf("stacked expert gate = %+v", gate)
	}
	router := candidate.Model.Tensors[routerName]
	if !tensor.HasDimensions(router.Layout, denseEmbeddingWidth, denseExpertCount) ||
		!slices.Equal(router.Values, candidate.Router.Layers[tensor.FirstOffset].Weights) {
		t.Fatalf("generated router = %+v", router)
	}
	if !slices.Equal(candidate.Model.Tensors[sharedTensorName].Values, []float32{31, 32}) {
		t.Fatal("generated model did not preserve the explicitly ordered base expert tensor")
	}
	if len(candidate.Lineage) == 0 {
		t.Fatal("generated candidate lacks lineage")
	}
}

func TestDenseToMoEExpertInventory(t *testing.T) {
	compiler := Compiler{}
	definition := testutil.ArtifactID(t, artifact.KindModelDefinition, "dense inventory definition")
	first := denseExpertSnapshot(t, compiler, definition, denseFirstExpertOffset)
	second := denseExpertSnapshot(t, compiler, definition, denseSecondExpertOffset)
	brokenWeights := cloneWeights(second.Tensors)
	brokenWeights[denseGateName] = Weight{
		Layout: tensor.MustShape(denseEmbeddingWidth, denseFeedForwardWidth+tensor.SingletonExtent),
		Values: make([]float32, denseEmbeddingWidth*(denseFeedForwardWidth+tensor.SingletonExtent)),
	}
	broken, err := compiler.Seal(definition, artifact.ID{}, brokenWeights)
	if err != nil {
		t.Fatal(err)
	}
	evidence := denseRouterEvidence(t, first.ID, broken.ID)
	if _, err := compiler.DenseToMoEArtifact(
		testutil.ArtifactID(t, artifact.KindRecipe, "dense inventory recipe"),
		testutil.ArtifactID(t, artifact.KindModelDefinition, "dense inventory target"),
		evidence, []Snapshot{first, broken}, denseLayerInventory(),
	); err == nil {
		t.Fatal("dense-to-MoE construction accepted a mismatched expert inventory")
	}
}

func TestDenseToMoEPromotionRefusal(t *testing.T) {
	candidate := denseToMoECandidateFixture(t)
	generation := promotedDenseGeneration(t, candidate)
	dataset := testutil.ArtifactID(t, artifact.KindDataset, "dense promotion dataset")
	baselineRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "dense promotion baseline recipe")
	baseline := denseEvaluation(t, baselineRecipe, dataset, denseBaselineQuality)
	worse := denseEvaluation(t, candidate.Screen.Recipe, dataset, denseWorseQuality)
	if _, err := candidate.promote(generation, baseline, worse); err == nil {
		t.Fatal("dense-to-MoE promotion accepted a held-out regression")
	}
	better := denseEvaluation(t, candidate.Screen.Recipe, dataset, denseImprovedQuality)
	promotion, err := candidate.promote(generation, baseline, better)
	if err != nil {
		t.Fatal(err)
	}
	if err := promotion.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
}

const (
	denseGateName           = "blk.0.ffn_gate.weight"
	denseUpName             = "blk.0.ffn_up.weight"
	denseDownName           = "blk.0.ffn_down.weight"
	routerName              = "blk.0.ffn_gate_inp.weight"
	expertGateName          = "blk.0.ffn_gate_exps.weight"
	expertUpName            = "blk.0.ffn_up_exps.weight"
	expertDownName          = "blk.0.ffn_down_exps.weight"
	sharedTensorName        = "output.weight"
	denseEmbeddingWidth     = 2
	denseFeedForwardWidth   = 3
	denseExpertCount        = 2
	denseFirstExpertOffset  = 0
	denseSecondExpertOffset = 10
	denseUpTensorOffset     = 21
	denseDownTensorOffset   = 41
	denseSharedTensorOffset = 31
	denseBaselineQuality    = 1
	denseWorseQuality       = 0.5
	denseImprovedQuality    = 1.5
)

func denseToMoECandidateFixture(t *testing.T) DenseToMoECandidate {
	t.Helper()
	compiler := Compiler{}
	definition := testutil.ArtifactID(t, artifact.KindModelDefinition, "dense source definition")
	first := denseExpertSnapshot(t, compiler, definition, denseFirstExpertOffset)
	second := denseExpertSnapshot(t, compiler, definition, denseSecondExpertOffset)
	candidate, err := compiler.DenseToMoEArtifact(
		testutil.ArtifactID(t, artifact.KindRecipe, "dense candidate recipe"),
		testutil.ArtifactID(t, artifact.KindModelDefinition, "dense target definition"),
		denseRouterEvidence(t, first.ID, second.ID), []Snapshot{first, second}, denseLayerInventory(),
	)
	if err != nil {
		t.Fatal(err)
	}
	return candidate
}

func denseExpertSnapshot(t *testing.T, compiler Compiler, definition artifact.ID, offset float32) Snapshot {
	t.Helper()
	sequence := func(count int, start float32) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = start + float32(index)
		}
		return values
	}
	snapshot, err := compiler.Seal(definition, artifact.ID{}, map[string]Weight{
		denseGateName: {
			Layout: tensor.MustShape(denseEmbeddingWidth, denseFeedForwardWidth),
			Values: sequence(denseEmbeddingWidth*denseFeedForwardWidth, offset+tensor.SingletonExtent),
		},
		denseUpName: {
			Layout: tensor.MustShape(denseEmbeddingWidth, denseFeedForwardWidth),
			Values: sequence(denseEmbeddingWidth*denseFeedForwardWidth, offset+denseUpTensorOffset),
		},
		denseDownName: {
			Layout: tensor.MustShape(denseFeedForwardWidth, denseEmbeddingWidth),
			Values: sequence(denseEmbeddingWidth*denseFeedForwardWidth, offset+denseDownTensorOffset),
		},
		sharedTensorName: {
			Layout: tensor.MustShape(denseEmbeddingWidth),
			Values: sequence(denseEmbeddingWidth, offset+denseSharedTensorOffset),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func denseRouterEvidence(t *testing.T, first, second artifact.ID) model.DenseToMoERouterEvidence {
	t.Helper()
	evidence, err := model.NewDenseToMoERouterEvidence(model.DenseToMoERouterEvidence{
		TrainingDataset: testutil.ArtifactID(t, artifact.KindDataset, "dense router training dataset"),
		HeldOutDataset:  testutil.ArtifactID(t, artifact.KindDataset, "dense router held-out dataset"),
		TrainingRun:     testutil.ArtifactID(t, artifact.KindRun, "dense router training run"),
		HeldOutRun:      testutil.ArtifactID(t, artifact.KindRun, "dense router held-out run"),
		Experts:         []artifact.ID{first, second},
		Training: []model.DenseToMoERouterSample{
			{Hidden: []float32{2, 0}, Experts: []artifact.ID{first}},
			{Hidden: []float32{0, 2}, Experts: []artifact.ID{second}},
		},
		HeldOut: []model.DenseToMoERouterSample{
			{Hidden: []float32{3, 0}, Experts: []artifact.ID{first}},
			{Hidden: []float32{0, 3}, Experts: []artifact.ID{second}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return evidence
}

func denseLayerInventory() []DenseToMoELayerTensors {
	return []DenseToMoELayerTensors{{
		Router: routerName, DenseGate: denseGateName, DenseUp: denseUpName, DenseDown: denseDownName,
		ExpertGate: expertGateName, ExpertUp: expertUpName, ExpertDown: expertDownName,
	}}
}

func promotedDenseGeneration(t *testing.T, candidate DenseToMoECandidate) evaluation.OfflineArtifactGenerationPromotion {
	t.Helper()
	names := make([]string, 0, len(candidate.Model.Tensors))
	values := make(map[string][]float32, len(candidate.Model.Tensors))
	for name, weight := range candidate.Model.Tensors {
		names = append(names, name)
		values[name] = weight.Values
	}
	slices.Sort(names)
	fixtures := make([]streamingFixtureTensor, len(names))
	for index, name := range names {
		fixtures[index] = streamingFixtureTensor{name: name, shape: []uint64{uint64(len(values[name]))}}
	}
	plan := sealStreamingPlan(
		t, composition.OfflineArtifactExactPassthrough,
		[]artifact.ID{candidate.Model.ID}, []float64{tensor.SingletonExtent}, fixtures,
	)
	evidence, err := composition.ValidateOfflineArtifactGeneration(
		plan, writeStreamingFixture(t, values), candidate.Model.ID,
		[]composition.OfflineArtifactGenerationTrial{{
			Seed:        tensor.SingletonExtent,
			Output:      testutil.ArtifactID(t, artifact.KindOutput, "dense generated output"),
			Run:         testutil.ArtifactID(t, artifact.KindRun, "dense generated run"),
			Observation: testutil.ArtifactID(t, artifact.KindEvidence, "dense generated observation"),
			LatencyNS:   tensor.SingletonExtent, PeakHostBytes: tensor.SingletonExtent,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := evaluation.NewOfflineArtifactGenerationPolicy(
		tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent,
		"unit fixture", "policy changes",
	)
	if err != nil {
		t.Fatal(err)
	}
	promotion, err := evaluation.PromoteOfflineArtifactGeneration(evidence, policy)
	if err != nil {
		t.Fatal(err)
	}
	return promotion
}

func denseEvaluation(t *testing.T, recipe, dataset artifact.ID, quality float64) runrecord.Evaluation {
	t.Helper()
	value, err := runrecord.NewEvaluation(
		recipe, testutil.ArtifactID(t, artifact.KindRun, fmt.Sprintf("dense evaluation run %s %g", recipe, quality)), dataset,
		[]runrecord.Metric{{Name: "quality", Value: quality, Direction: runrecord.DirectionMaximize}},
	)
	if err != nil {
		t.Fatal(err)
	}
	return value
}
