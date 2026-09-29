package model

import (
	"strings"
	"testing"
)

func requireLayerProgram(t *testing.T, program layerProgram, want ...layerOperator) {
	t.Helper()
	if len(program.Instructions) != len(want) {
		t.Fatalf("stage count = %d, want %d", len(program.Instructions), len(want))
	}
	for index, operator := range want {
		instruction, ok := program.Instruction(uint32(index))
		if !ok || instruction.Operator != operator {
			t.Fatalf("stage %d = %+v, want operator %d", index, instruction, operator)
		}
	}
}

func mustCompileLayerProgram(t *testing.T, plan LayerPlan, profile ArchitectureProfile) layerProgram {
	t.Helper()
	program, err := compileLayerProgram(plan, profile)
	if err != nil {
		t.Fatal(err)
	}
	return program
}

func typedFixtureLayerProgram(t *testing.T, profile ArchitectureProfile, plan LayerPlan) layerProgram {
	t.Helper()
	plan.Attention = profile.Attention
	plan.Mixer = compileRecurrentMixer(profile, plan.Recurrent)
	return mustCompileLayerProgram(t, plan, profile)
}

func TestLayerProgramsCoverTypedProfiles(t *testing.T) {
	attentionNorm := []layerOperator{
		layerOperatorAttentionNorm, layerOperatorLatentAttention, layerOperatorResidual,
		layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy, layerOperatorResidual,
	}
	recurrent := []layerOperator{layerOperatorAttentionNorm, layerOperatorRecurrentMix, layerOperatorResidual}
	tests := []struct {
		name    string
		profile ArchitectureProfile
		plan    LayerPlan
		want    []layerOperator
	}{
		{name: "dense", want: []layerOperator{
			layerOperatorAttentionInputNorm, layerOperatorAttentionPlannedProjection, layerOperatorResidual,
			layerOperatorFeedForwardInputNorm, layerOperatorFeedForwardPolicy,
			layerOperatorFeedForwardOutput, layerOperatorResidual,
		}},
		{name: "keyed delta", profile: ArchitectureProfile{LayerTopology: layerTopologyKeyedDeltaHybrid, RecurrentMixer: recurrentMixerKeyedDelta}, want: attentionNorm},
		{name: "MLA", profile: ArchitectureProfile{Attention: attentionLatent}, want: attentionNorm},
		{name: "DSA", profile: ArchitectureProfile{Attention: AttentionSparseLatent}, want: attentionNorm},
		{name: "compressed hyper", profile: ArchitectureProfile{LayerTopology: layerTopologyCompressedHyper}, want: []layerOperator{
			layerOperatorHyperAttention, layerOperatorHyperFeedForward,
		}},
		{name: "selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerSelectiveScan}, want: recurrent},
		{name: "grouped selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerGroupedSelectiveScan}, want: recurrent},
		{name: "weighted selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerWeightedSelectiveScan}, plan: LayerPlan{Recurrent: true}, want: append(recurrent, layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy, layerOperatorResidual)},
		{name: "scaled grouped selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerScaledGroupedSelectiveScan}, plan: LayerPlan{Recurrent: true}, want: []layerOperator{
			layerOperatorAttentionNorm, layerOperatorRecurrentMix, layerOperatorScale,
			layerOperatorResidual, layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy,
			layerOperatorScale, layerOperatorResidual,
		}},
		{name: "normalized selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerNormalizedSelectiveScan}, plan: LayerPlan{Recurrent: true}, want: []layerOperator{
			layerOperatorAttentionNorm, layerOperatorRecurrentMix, layerOperatorAttentionPostNorm,
			layerOperatorResidual, layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy,
			layerOperatorFeedForwardPostNorm, layerOperatorResidual,
		}},
		{name: "sparse grouped selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerSparseGroupedSelectiveScan}, plan: LayerPlan{Composition: layerCompositionAttentionOnly}, want: []layerOperator{
			layerOperatorAttentionNorm, layerOperatorAttentionCausalProjection, layerOperatorResidual,
		}},
		{name: "attention grouped selective scan", profile: ArchitectureProfile{RecurrentMixer: recurrentMixerAttentionGroupedSelectiveScan}, want: []layerOperator{
			layerOperatorAttentionNorm, layerOperatorHybridMix, layerOperatorResidual,
			layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy, layerOperatorResidual,
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
		LayerPlan{}, ArchitectureProfile{LayerTopology: layerTopologySharedKVAdapter},
	)
	want := []layerOperator{
		layerOperatorAttentionNorm, layerOperatorAttentionSharedKVQKNorm, layerOperatorAttentionPostNorm,
		layerOperatorResidual, layerOperatorFeedForwardParallelGatedGELU, layerOperatorResidual,
		layerOperatorOutputAdapter,
	}
	requireLayerProgram(t, program, want...)
}

func TestEagle3ProgramUsesPairedInputStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
		LayerPlan{}, ArchitectureProfile{Forward: ForwardProgram{Operation: ForwardOperationSession, Session: ForwardSessionFeatureDraft}},
	)
	want := []layerOperator{
		layerOperatorPairedInputNorm, layerOperatorAttentionPairedCausalProjection, layerOperatorResidual,
		layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy, layerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
	paired, _ := program.Instruction(0)
	if paired.Tensors[0] != runtimeTensorPerLayerInput {
		t.Fatalf("Eagle3 paired binding = %+v", paired)
	}
}

func TestGemma4AssistantProgramUsesSharedCacheStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
		LayerPlan{}, ArchitectureProfile{Forward: ForwardProgram{Operation: ForwardOperationSession, Session: ForwardSessionPairedProjection}},
	)
	want := []layerOperator{
		layerOperatorAttentionNorm, layerOperatorAttentionSharedCacheQKNorm, layerOperatorAttentionPostNorm,
		layerOperatorResidual, layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy,
		layerOperatorFeedForwardPostNorm, layerOperatorResidualScale,
	}
	requireLayerProgram(t, program, want...)
	attention, _ := program.Instruction(1)
	if attention.Operator != layerOperatorAttentionSharedCacheQKNorm ||
		attention.Caches[0] != runtimeCachePrimaryKey || attention.Caches[1] != runtimeCachePrimaryValue {
		t.Fatalf("Gemma 4 assistant attention binding = %+v", attention)
	}
}

func TestRWKVProgramsUseNeutralStages(t *testing.T) {
	tests := []struct {
		name    string
		profile ArchitectureProfile
		want    []layerOperator
	}{
		{
			name: "WKV6", profile: ArchitectureProfile{LayerTopology: layerTopologyDynamicWKV6},
			want: []layerOperator{
				layerOperatorAttentionNorm, layerOperatorRecurrentMix, layerOperatorResidual,
				layerOperatorGatedTokenShiftSquaredReLU, layerOperatorResidual,
			},
		},
		{
			name: "WKV7 channel", profile: ArchitectureProfile{
				LayerTopology: layerTopologyDynamicWKV7, Normalization: normalizationLayer,
			},
			want: []layerOperator{
				layerOperatorAttentionNorm, layerOperatorRecurrentMix, layerOperatorResidual,
				layerOperatorTokenShiftSquaredReLU, layerOperatorResidual,
			},
		},
		{
			name: "WKV7 SwiGLU", profile: ArchitectureProfile{
				LayerTopology: layerTopologyDynamicWKV7, Normalization: NormalizationRMS,
			},
			want: []layerOperator{
				layerOperatorAttentionNorm, layerOperatorRecurrentMix, layerOperatorResidual,
				layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy, layerOperatorResidual,
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
		ArchitectureProfile{LayerTopology: layerTopologyDynamicWKV6},
	)
	requireLayerProgram(t, periodic,
		layerOperatorAttentionNorm, layerOperatorRecurrentMix, layerOperatorResidual,
		layerOperatorGatedTokenShiftSquaredReLU, layerOperatorResidual, layerOperatorScale,
	)
}

func TestTalkieProgramUsesNeutralStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
		LayerPlan{}, ArchitectureProfile{LayerTopology: LayerTopologyCausalPostQKNormSkip},
	)
	want := []layerOperator{
		layerOperatorRMSNorm, layerOperatorAttentionCausalPostQKNorm, layerOperatorResidual,
		layerOperatorRMSNorm, layerOperatorFeedForwardPolicy, layerOperatorResidual,
		layerOperatorScaledSkip,
	}
	requireLayerProgram(t, program, want...)
}

func TestBERTProgramUsesPostNormalizedStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
		LayerPlan{}, ArchitectureProfile{LayerTopology: layerTopologyBidirectionalEncoder},
	)
	want := []layerOperator{
		layerOperatorAttentionBidirectionalEncoder, layerOperatorAttentionResidualNorm,
		layerOperatorFeedForwardEncoder, layerOperatorFeedForwardResidualNorm,
	}
	requireLayerProgram(t, program, want...)
}

func TestJinaV2ProgramAddsInputResidualNormalization(t *testing.T) {
	program := mustCompileLayerProgram(t, LayerPlan{}, ArchitectureProfile{
		LayerTopology:   layerTopologyBidirectionalEncoder,
		EncoderOperator: encoderOperatorPostNormALiBi,
	})
	requireLayerProgram(t, program,
		layerOperatorAttentionBidirectionalEncoder,
		layerOperatorAttentionResidualNorm,
		layerOperatorInputResidualNorm,
		layerOperatorFeedForwardEncoder,
		layerOperatorFeedForwardResidualNorm,
	)
}

func TestGemmaEmbeddingProgramUsesNeutralStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
		LayerPlan{}, ArchitectureProfile{LayerTopology: layerTopologyBidirectionalQKNorm},
	)
	want := []layerOperator{
		layerOperatorAttentionNorm, layerOperatorAttentionBidirectionalQKNorm, layerOperatorAttentionPostNorm,
		layerOperatorResidual, layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy,
		layerOperatorFeedForwardPostNorm, layerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
}

func TestDeciSparseProgramUsesNeutralStages(t *testing.T) {
	program := mustCompileLayerProgram(t, LayerPlan{DeciSparse: true}, ArchitectureProfile{})
	want := []layerOperator{
		layerOperatorCacheSentinel, layerOperatorAttentionNorm, layerOperatorAttentionOutputProjection,
		layerOperatorResidual, layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy,
		layerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
}

func TestKeyedDeltaProgramSelectsLinearAttention(t *testing.T) {
	profile := ArchitectureProfile{LayerTopology: layerTopologyKeyedDeltaHybrid, RecurrentMixer: recurrentMixerKeyedDelta}
	program := typedFixtureLayerProgram(t, profile, LayerPlan{Recurrent: true})
	requireLayerProgram(t, program,
		layerOperatorAttentionNorm, layerOperatorRecurrentMix, layerOperatorResidual,
		layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy, layerOperatorResidual,
	)
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	state := compileRecurrentMixer(profile, true)
	if mixer.Operator != layerOperatorRecurrentMix || state != recurrentMixerKeyedDelta ||
		feedForward.Operator != layerOperatorFeedForwardPolicy {
		t.Fatalf("keyed-delta policies = %+v/%+v/%d", mixer, feedForward, state)
	}
}

func TestLFM2RecurrentProgramUsesSharedStages(t *testing.T) {
	program := mustCompileLayerProgram(t,
		LayerPlan{Recurrent: true},
		ArchitectureProfile{Attention: attentionShortConvolution, RecurrentMixer: recurrentMixerShortConvolution},
	)
	want := []layerOperator{
		layerOperatorAttentionNorm, layerOperatorRecurrentMix, layerOperatorResidual,
		layerOperatorFeedForwardNorm, layerOperatorFeedForwardPolicy, layerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	state := compileRecurrentMixer(
		ArchitectureProfile{RecurrentMixer: recurrentMixerShortConvolution}, true,
	)
	if mixer.Operator != layerOperatorRecurrentMix || state != recurrentMixerShortConvolution ||
		feedForward.Operator != layerOperatorFeedForwardPolicy {
		t.Fatalf("LFM2 recurrent policies = %+v/%d/%d", mixer, state, feedForward.Operator)
	}
}

func TestNemotronLayerProgramsSelectSemanticMixer(t *testing.T) {
	const mixerStage = 1
	fixtures := []struct {
		name        string
		recurrent   bool
		composition layerCompositionPolicy
		mixer       layerOperator
	}{
		{name: "recurrent", recurrent: true, composition: layerCompositionRecurrentOnly, mixer: layerOperatorRecurrentMix},
		{name: "attention", composition: layerCompositionAttentionOnly, mixer: layerOperatorAttentionCausalProjection},
		{name: "feed-forward", composition: layerCompositionFeedForwardOnly, mixer: layerOperatorCacheSentinel},
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
	for _, recurrent := range []bool{false, true} {
		mixer := layerOperatorAttentionGatedProjection
		if recurrent {
			mixer = layerOperatorRecurrentMix
		}
		program := mustCompileLayerProgram(t,
			LayerPlan{Recurrent: recurrent},
			ArchitectureProfile{Attention: AttentionGatedDelta, RecurrentMixer: recurrentMixerGatedDelta},
		)
		requireLayerProgram(t, program,
			layerOperatorAttentionNorm, mixer, layerOperatorResidual,
			layerOperatorFeedForwardNorm, layerOperatorFeedForwardRoutedSwiGLU, layerOperatorResidual,
		)
		state := compileRecurrentMixer(
			ArchitectureProfile{RecurrentMixer: recurrentMixerGatedDelta}, recurrent,
		)
		if recurrent && state != recurrentMixerGatedDelta {
			t.Fatalf("Qwen recurrent mixer = %v", state)
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
