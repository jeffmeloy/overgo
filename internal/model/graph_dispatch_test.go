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

func mustCompileLayerProgram(t *testing.T, plan LayerPlan, profile ArchitectureProfile) LayerProgram {
	t.Helper()
	program, err := compileLayerProgram(plan, profile)
	if err != nil {
		t.Fatal(err)
	}
	return program
}

func typedFixtureLayerProgram(t *testing.T, profile ArchitectureProfile, plan LayerPlan) LayerProgram {
	t.Helper()
	plan.Attention = profile.Attention
	plan.Mixer = compileRecurrentMixer(profile, plan.Recurrent)
	return mustCompileLayerProgram(t, plan, profile)
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
			requireLayerProgram(t, typedFixtureLayerProgram(t, test.profile, test.plan), test.want...)
		})
	}
}

func TestGemma4ProgramUsesNeutralStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
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
	program := mustCompileLayerProgram(t,
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
	program := mustCompileLayerProgram(t,
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
				LayerOperatorGatedTokenShiftSquaredReLU, LayerOperatorResidual,
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
			program := mustCompileLayerProgram(t, LayerPlan{}, test.profile)
			requireLayerProgram(t, program, test.want...)
		})
	}
	periodic := mustCompileLayerProgram(t,
		LayerPlan{PeriodicScale: 0.5},
		ArchitectureProfile{LayerTopology: LayerTopologyDynamicWKV6},
	)
	requireLayerProgram(t, periodic,
		LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
		LayerOperatorGatedTokenShiftSquaredReLU, LayerOperatorResidual, LayerOperatorPeriodicScale,
	)
}

func TestTalkieProgramUsesNeutralStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
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
	program := mustCompileLayerProgram(t,
		LayerPlan{}, ArchitectureProfile{LayerTopology: LayerTopologyBidirectionalEncoder},
	)
	want := []LayerOperator{
		LayerOperatorAttentionBidirectionalEncoder, LayerOperatorAttentionResidualNorm,
		LayerOperatorFeedForwardEncoder, LayerOperatorFeedForwardResidualNorm,
	}
	requireLayerProgram(t, program, want...)
}

func TestJinaV2ProgramAddsInputResidualNormalization(t *testing.T) {
	program := mustCompileLayerProgram(t, LayerPlan{}, ArchitectureProfile{
		LayerTopology:   LayerTopologyBidirectionalEncoder,
		EncoderOperator: encoderOperatorPostNormALiBi,
	})
	instruction, ok := program.Instruction(2)
	if !ok || program.Count != 5 || instruction.Operator != LayerOperatorInputResidualNorm {
		t.Fatalf("JinaV2 program = %+v", program)
	}
}

func TestGemmaEmbeddingProgramUsesNeutralStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
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
	program := mustCompileLayerProgram(t, LayerPlan{DeciSparse: true}, ArchitectureProfile{})
	want := []LayerOperator{
		LayerOperatorCacheSentinel, LayerOperatorAttentionNorm, LayerOperatorAttentionOutputProjection,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU,
		LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
}

func TestKeyedDeltaProgramSelectsLinearAttention(t *testing.T) {
	profile := ArchitectureProfile{LayerTopology: LayerTopologyKeyedDeltaHybrid, RecurrentMixer: recurrentMixerKeyedDelta}
	program := typedFixtureLayerProgram(t, profile, LayerPlan{Recurrent: true})
	requireLayerProgram(t, program,
		LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
	)
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	state := compileRecurrentMixer(profile, true)
	if mixer.Operator != LayerOperatorRecurrentMix || state != recurrentMixerKeyedDelta ||
		feedForward.Operator != LayerOperatorFeedForwardStandardSwiGLU {
		t.Fatalf("keyed-delta policies = %+v/%+v/%d", mixer, feedForward, state)
	}
}

func TestLFM2RecurrentProgramUsesSharedStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
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
	state := compileRecurrentMixer(
		ArchitectureProfile{RecurrentMixer: recurrentMixerShortConvolution}, true,
	)
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
			program := typedFixtureLayerProgram(t, profile, LayerPlan{
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
		program := mustCompileLayerProgram(t,
			LayerPlan{Recurrent: recurrent},
			ArchitectureProfile{Attention: AttentionGatedDelta, RecurrentMixer: recurrentMixerGatedDelta},
		)
		instruction, ok := program.Instruction(mixerStage)
		if !ok || program.Count != qwenProgramStageCount {
			t.Fatalf("Qwen GDN program = %+v", program)
		}
		state := compileRecurrentMixer(
			ArchitectureProfile{RecurrentMixer: recurrentMixerGatedDelta}, recurrent,
		)
		if recurrent && (instruction.Operator != LayerOperatorRecurrentMix || state != recurrentMixerGatedDelta) {
			t.Fatalf("Qwen recurrent stage = %+v", instruction)
		}
		if !recurrent && instruction.Operator != LayerOperatorAttentionGatedProjection {
			t.Fatalf("Qwen attention stage = %+v", instruction)
		}
	}
}

func TestCompiledBlockDispatchRequiresPlan(t *testing.T) {
	_, err := executeCompiledLayer(blockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: "unknown"}},
	})
	if err == nil || !strings.Contains(err.Error(), "compiled layer plan is required") {
		t.Fatalf("unknown error = %v", err)
	}
	llama, _ := LookupArchitecture("llama")
	_, err = executeCompiledLayer(blockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: llama.Name}}.withProfile(llama),
	})
	if err == nil || !strings.Contains(err.Error(), "compiled layer plan is required") {
		t.Fatalf("missing plan error = %v", err)
	}
}
