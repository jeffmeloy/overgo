package model

import (
	"strings"
	"testing"
)

func requireLayerProgram(t *testing.T, program LayerProgram, want ...LayerOperator) {
	t.Helper()
	if program.Count != uint8(len(want)) {
		t.Fatalf("stage count = %d, want %d", program.Count, len(want))
	}
	for index, operator := range want {
		instruction, ok := program.Instruction(index)
		if !ok || instruction.Operator != operator {
			t.Fatalf("stage %d = %+v, want operator %d", index, instruction, operator)
		}
	}
}

func typedFixtureLayerProgram(profile ArchitectureProfile, plan LayerPlan) LayerProgram {
	plan.Attention = profile.Attention
	plan.Mixer = Spec{}.withProfile(profile).compileRecurrentMixer(plan.Recurrent)
	return compileLayerProgram(plan, profile)
}

func TestLayerProgramsCoverTypedProfiles(t *testing.T) {
	attentionNorm := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorLatentAttention, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
	}
	recurrent := []LayerOperator{LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual}
	tests := []struct {
		name    string
		profile ArchitectureProfile
		plan    LayerPlan
		want    []LayerOperator
	}{
		{name: "dense", want: []LayerOperator{
			LayerOperatorAttentionInputNorm, LayerOperatorAttentionPlannedProjection, LayerOperatorResidual,
			LayerOperatorFeedForwardInputNorm, LayerOperatorFeedForwardPlanned,
			LayerOperatorFeedForwardOutput, LayerOperatorResidual,
		}},
		{name: "keyed delta", profile: ArchitectureProfile{LayerTopology: LayerTopologyKeyedDeltaHybrid, RecurrentMixer: recurrentMixerKeyedDelta}, want: attentionNorm},
		{name: "MLA", profile: ArchitectureProfile{Attention: AttentionLatent}, want: attentionNorm},
		{name: "DSA", profile: ArchitectureProfile{Attention: AttentionSparseLatent}, want: attentionNorm},
		{name: "compressed hyper", profile: ArchitectureProfile{LayerTopology: LayerTopologyCompressedHyper}, want: []LayerOperator{
			LayerOperatorHyperAttention, LayerOperatorHyperFeedForward,
		}},
		{name: "selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerSelectiveScan}, want: recurrent},
		{name: "grouped selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerGroupedSelectiveScan}, want: recurrent},
		{name: "weighted selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerWeightedSelectiveScan}, plan: LayerPlan{Recurrent: true}, want: append(recurrent, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual)},
		{name: "scaled grouped selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerScaledGroupedSelectiveScan}, plan: LayerPlan{Recurrent: true}, want: []LayerOperator{
			LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorScale,
			LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU,
			LayerOperatorScale, LayerOperatorResidual,
		}},
		{name: "normalized selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerNormalizedSelectiveScan}, plan: LayerPlan{Recurrent: true}, want: []LayerOperator{
			LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorAttentionPostNorm,
			LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardFusedGLU,
			LayerOperatorFeedForwardPostNorm, LayerOperatorResidual,
		}},
		{name: "sparse grouped selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerSparseGroupedSelectiveScan}, plan: LayerPlan{Composition: LayerCompositionAttentionOnly}, want: []LayerOperator{
			LayerOperatorAttentionNorm, LayerOperatorAttentionCausalProjection, LayerOperatorResidual,
		}},
		{name: "attention grouped selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerAttentionGroupedSelectiveScan}, want: []LayerOperator{
			LayerOperatorAttentionNorm, LayerOperatorHybridMix, LayerOperatorResidual,
			LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requireLayerProgram(t, typedFixtureLayerProgram(test.profile, test.plan), test.want...)
		})
	}
}

func TestGemma4ProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{}, ArchitectureProfile{LayerTopology: LayerTopologySharedKVAdapter},
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorAttentionSharedKVQKNorm, LayerOperatorAttentionPostNorm,
		LayerOperatorResidual, LayerOperatorFeedForwardParallelGatedGELU, LayerOperatorResidual,
		LayerOperatorOutputAdapter,
	}
	requireLayerProgram(t, program, want...)
}

func TestEagle3ProgramUsesPairedInputStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{}, ArchitectureProfile{Forward: ForwardProgram{Operation: ForwardOperationSession, Session: ForwardSessionFeatureDraft}},
	)
	want := []LayerOperator{
		LayerOperatorPairedInputNorm, LayerOperatorAttentionPairedCausalProjection, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
	paired, _ := program.Instruction(0)
	if paired.TensorCount != 1 || paired.Tensors[0] != RuntimeTensorPerLayerInput {
		t.Fatalf("Eagle3 paired binding = %+v", paired)
	}
}

func TestGemma4AssistantProgramUsesSharedCacheStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{}, ArchitectureProfile{Forward: ForwardProgram{Operation: ForwardOperationSession, Session: ForwardSessionPairedProjection}},
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorAttentionSharedCacheQKNorm, LayerOperatorAttentionPostNorm,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardGatedGELU,
		LayerOperatorFeedForwardPostNorm, LayerOperatorResidualScale,
	}
	requireLayerProgram(t, program, want...)
	attention, _ := program.Instruction(1)
	if attention.Operator != LayerOperatorAttentionSharedCacheQKNorm || attention.CacheCount != 2 {
		t.Fatalf("Gemma 4 assistant attention binding = %+v", attention)
	}
}

func TestRWKVProgramsUseNeutralStages(t *testing.T) {
	tests := []struct {
		name    string
		profile ArchitectureProfile
		want    []LayerOperator
	}{
		{
			name: "WKV6", profile: ArchitectureProfile{LayerTopology: LayerTopologyDynamicWKV6},
			want: []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
				LayerOperatorGatedTokenShiftSquaredReLU, LayerOperatorResidual, LayerOperatorPeriodicScale,
			},
		},
		{
			name: "WKV7 channel", profile: ArchitectureProfile{
				LayerTopology: LayerTopologyDynamicWKV7, Normalization: NormalizationLayer,
			},
			want: []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
				LayerOperatorTokenShiftSquaredReLU, LayerOperatorResidual,
			},
		},
		{
			name: "WKV7 SwiGLU", profile: ArchitectureProfile{
				LayerTopology: LayerTopologyDynamicWKV7, Normalization: NormalizationRMS,
			},
			want: []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			program := compileLayerProgram(LayerPlan{}, test.profile)
			requireLayerProgram(t, program, test.want...)
		})
	}
}

func TestTalkieProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{}, ArchitectureProfile{LayerTopology: LayerTopologyCausalPostQKNormSkip},
	)
	want := []LayerOperator{
		LayerOperatorRMSNorm, LayerOperatorAttentionCausalPostQKNorm, LayerOperatorResidual,
		LayerOperatorRMSNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
		LayerOperatorScaledSkip,
	}
	requireLayerProgram(t, program, want...)
}

func TestBERTProgramUsesPostNormalizedStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{}, ArchitectureProfile{LayerTopology: LayerTopologyBidirectionalEncoder},
	)
	want := []LayerOperator{
		LayerOperatorAttentionBidirectionalEncoder, LayerOperatorAttentionResidualNorm,
		LayerOperatorFeedForwardEncoder, LayerOperatorFeedForwardResidualNorm,
	}
	requireLayerProgram(t, program, want...)
}

func TestJinaV2ProgramAddsInputResidualNormalization(t *testing.T) {
	program := compileLayerProgram(LayerPlan{}, ArchitectureProfile{
		LayerTopology:   LayerTopologyBidirectionalEncoder,
		EncoderOperator: encoderOperatorPostNormALiBi,
	})
	instruction, ok := program.Instruction(2)
	if !ok || program.Count != 5 || instruction.Operator != LayerOperatorInputResidualNorm {
		t.Fatalf("JinaV2 program = %+v", program)
	}
}

func TestGemmaEmbeddingProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{}, ArchitectureProfile{LayerTopology: LayerTopologyBidirectionalQKNorm},
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorAttentionBidirectionalQKNorm, LayerOperatorAttentionPostNorm,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardGatedGELU,
		LayerOperatorFeedForwardPostNorm, LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
}

func TestDeciSparseProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(LayerPlan{DeciSparse: true}, ArchitectureProfile{})
	want := []LayerOperator{
		LayerOperatorCacheSentinel, LayerOperatorAttentionNorm, LayerOperatorAttentionOutputProjection,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU,
		LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
}

func TestKeyedDeltaProgramSelectsLinearAttention(t *testing.T) {
	profile := ArchitectureProfile{LayerTopology: LayerTopologyKeyedDeltaHybrid, RecurrentMixer: recurrentMixerKeyedDelta}
	program := typedFixtureLayerProgram(profile, LayerPlan{Recurrent: true})
	requireLayerProgram(t, program,
		LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
	)
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	state := Spec{RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true}}}.
		withProfile(profile).
		compileRecurrentMixer(true)
	if mixer.Operator != LayerOperatorRecurrentMix || state != recurrentMixerKeyedDelta ||
		feedForward.Operator != LayerOperatorFeedForwardStandardSwiGLU {
		t.Fatalf("keyed-delta policies = %+v/%+v/%d", mixer, feedForward, state)
	}
}

func TestLFM2RecurrentProgramUsesSharedStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Recurrent: true},
		ArchitectureProfile{Attention: AttentionShortConvolution, RecurrentMixer: recurrentMixerShortConvolution},
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	state := Spec{RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true}}}.
		withProfile(ArchitectureProfile{RecurrentMixer: recurrentMixerShortConvolution}).compileRecurrentMixer(true)
	if mixer.Operator != LayerOperatorRecurrentMix || state != recurrentMixerShortConvolution ||
		feedForward.Operator != LayerOperatorFeedForwardStandardSwiGLU {
		t.Fatalf("LFM2 recurrent policies = %+v/%d/%d", mixer, state, feedForward.Operator)
	}
}

func TestNemotronLayerProgramsSelectSemanticMixer(t *testing.T) {
	const mixerStage = 1
	fixtures := []struct {
		name        string
		recurrent   bool
		composition LayerCompositionPolicy
		mixer       LayerOperator
	}{
		{name: "recurrent", recurrent: true, composition: LayerCompositionRecurrentOnly, mixer: LayerOperatorRecurrentMix},
		{name: "attention", composition: LayerCompositionAttentionOnly, mixer: LayerOperatorAttentionCausalProjection},
		{name: "feed-forward", composition: LayerCompositionFeedForwardOnly, mixer: LayerOperatorCacheSentinel},
	}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			profile := ArchitectureProfile{RecurrentMixer: recurrentMixerSparseGroupedSelectiveScan}
			program := typedFixtureLayerProgram(profile, LayerPlan{
				Recurrent: fixture.recurrent, Composition: fixture.composition,
			})
			instruction, ok := program.Instruction(mixerStage)
			if !ok || instruction.Operator != fixture.mixer {
				t.Fatalf("Nemotron program = %+v", program)
			}
		})
	}
}

func TestQwenGDNProgramsSelectSemanticMixer(t *testing.T) {
	const (
		mixerStage            = 1
		qwenProgramStageCount = 6
	)
	for _, recurrent := range []bool{false, true} {
		program := compileLayerProgram(
			LayerPlan{Recurrent: recurrent},
			ArchitectureProfile{Attention: AttentionGatedDelta, RecurrentMixer: recurrentMixerGatedDelta},
		)
		instruction, ok := program.Instruction(mixerStage)
		if !ok || program.Count != qwenProgramStageCount {
			t.Fatalf("Qwen GDN program = %+v", program)
		}
		state := Spec{RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{recurrent}}}.
			withProfile(ArchitectureProfile{RecurrentMixer: recurrentMixerGatedDelta}).compileRecurrentMixer(recurrent)
		if recurrent && (instruction.Operator != LayerOperatorRecurrentMix || state != recurrentMixerGatedDelta) {
			t.Fatalf("Qwen recurrent stage = %+v", instruction)
		}
		if !recurrent && instruction.Operator != LayerOperatorAttentionGatedProjection {
			t.Fatalf("Qwen attention stage = %+v", instruction)
		}
	}
}

func TestCompiledBlockDispatchRequiresPlan(t *testing.T) {
	_, err := executeCompiledLayer(BlockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: "unknown"}},
	})
	if err == nil || !strings.Contains(err.Error(), "compiled layer plan is required") {
		t.Fatalf("unknown error = %v", err)
	}
	t5 := Spec{CommonSpec: CommonSpec{Architecture: "t5"}}
	t5Plan := t5.PlanLayer(0, false)
	_, err = executeCompiledLayer(BlockDispatchOptions{Spec: t5, Plan: &t5Plan})
	if err == nil || !strings.Contains(err.Error(), "explicit encoder state") {
		t.Fatalf("T5 dispatch error = %v", err)
	}
	external := ArchitectureProfile{
		Name:    "external-encoder-decoder",
		Forward: ForwardProgram{Operation: ForwardOperationSession, Session: ForwardSessionEncoderDecoder},
	}
	externalSpec := Spec{CommonSpec: CommonSpec{Architecture: external.Name}}.withProfile(external)
	externalPlan := externalSpec.PlanLayer(0, false)
	_, err = executeCompiledLayer(BlockDispatchOptions{
		Spec: externalSpec, Plan: &externalPlan,
	})
	if err == nil || !strings.Contains(err.Error(), "explicit encoder state") {
		t.Fatalf("bound external dispatch error = %v", err)
	}
	llama, _ := LookupArchitecture("llama")
	_, err = executeCompiledLayer(BlockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: llama.Name}}.withProfile(llama),
	})
	if err == nil || !strings.Contains(err.Error(), "compiled layer plan is required") {
		t.Fatalf("missing plan error = %v", err)
	}
}
