package model

import (
	"errors"
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

func TestLayerProgramsCoverCompiledPolicies(t *testing.T) {
	for policy := BlockDense; policy <= BlockDeepSeek4; policy++ {
		composition := LayerCompositionStandard
		if policy == BlockNemotronH {
			composition = LayerCompositionAttentionOnly
		}
		program := compileLayerProgram(
			LayerPlan{Composition: composition}, ArchitectureProfile{}, policy,
		)
		var want []LayerOperator
		switch policy {
		case BlockDense:
			want = []LayerOperator{
				LayerOperatorAttentionInputNorm, LayerOperatorAttentionPlannedProjection, LayerOperatorResidual,
				LayerOperatorFeedForwardInputNorm, LayerOperatorFeedForwardPlanned,
				LayerOperatorFeedForwardOutput, LayerOperatorResidual,
			}
		case BlockKimiLinear:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorLatentAttention, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
			}
		case BlockMLA:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorLatentAttention, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
			}
		case BlockDSA:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorLatentAttention, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
			}
		case BlockDeepSeek4:
			want = []LayerOperator{
				LayerOperatorHyperAttention, LayerOperatorHyperFeedForward,
			}
		case BlockMamba, BlockMamba2:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
			}
		case BlockJamba:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
			}
		case BlockGraniteHybrid:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorScale,
				LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU,
				LayerOperatorScale, LayerOperatorResidual,
			}
		case BlockPLaMo2:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorAttentionPostNorm,
				LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardFusedGLU,
				LayerOperatorFeedForwardPostNorm, LayerOperatorResidual,
			}
		case BlockNemotronH:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorAttentionCausalProjection, LayerOperatorResidual,
			}
		case BlockFalconH1:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorHybridMix, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
			}
		}
		requireLayerProgram(t, program, want...)
	}
}

func TestGemma4ProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{}, ArchitectureProfile{DenseGraph: DenseGraphGemma4}, BlockDense,
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
		LayerPlan{}, ArchitectureProfile{Forward: ForwardEagle3}, BlockDense,
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
		LayerPlan{}, ArchitectureProfile{Forward: ForwardGemma4Assistant}, BlockDense,
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
			name: "RWKV6", profile: ArchitectureProfile{DenseGraph: DenseGraphRWKV6},
			want: []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
				LayerOperatorGatedTokenShiftSquaredReLU, LayerOperatorResidual, LayerOperatorPeriodicScale,
			},
		},
		{
			name: "RWKV7 channel", profile: ArchitectureProfile{
				DenseGraph: DenseGraphRWKV7, Normalization: NormalizationLayer,
			},
			want: []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
				LayerOperatorTokenShiftSquaredReLU, LayerOperatorResidual,
			},
		},
		{
			name: "RWKV7 SwiGLU", profile: ArchitectureProfile{
				DenseGraph: DenseGraphRWKV7, Normalization: NormalizationRMS,
			},
			want: []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			program := compileLayerProgram(LayerPlan{}, test.profile, BlockDense)
			requireLayerProgram(t, program, test.want...)
		})
	}
}

func TestTalkieProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{}, ArchitectureProfile{DenseGraph: DenseGraphTalkie}, BlockDense,
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
		LayerPlan{}, ArchitectureProfile{DenseGraph: DenseGraphBERT}, BlockDense,
	)
	want := []LayerOperator{
		LayerOperatorAttentionBidirectionalEncoder, LayerOperatorAttentionResidualNorm,
		LayerOperatorFeedForwardEncoder, LayerOperatorFeedForwardResidualNorm,
	}
	requireLayerProgram(t, program, want...)
}

func TestJinaV2ProgramAddsInputResidualNormalization(t *testing.T) {
	program := compileLayerProgram(LayerPlan{}, ArchitectureProfile{
		DenseGraph: DenseGraphBERT, EncoderGraph: EncoderGraphPolicy{Kind: encoderGraphJinaV2},
	}, BlockDense)
	instruction, ok := program.Instruction(2)
	if !ok || program.Count != 5 || instruction.Operator != LayerOperatorInputResidualNorm {
		t.Fatalf("JinaV2 program = %+v", program)
	}
}

func TestGemmaEmbeddingProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{}, ArchitectureProfile{DenseGraph: DenseGraphGemmaEmbedding}, BlockDense,
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorAttentionBidirectionalQKNorm, LayerOperatorAttentionPostNorm,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardGatedGELU,
		LayerOperatorFeedForwardPostNorm, LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
}

func TestDeciSparseProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(LayerPlan{DeciSparse: true}, ArchitectureProfile{}, BlockDense)
	want := []LayerOperator{
		LayerOperatorCacheSentinel, LayerOperatorAttentionNorm, LayerOperatorAttentionOutputProjection,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU,
		LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
}

func TestKimiRecurrentProgramSelectsLinearAttention(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Recurrent: true}, ArchitectureProfile{}, BlockKimiLinear,
	)
	requireLayerProgram(t, program,
		LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
	)
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	state := Spec{RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true}}}.
		withProfile(ArchitectureProfile{Block: BlockKimiLinear}).stateSpacePlan(0, true)
	if mixer.Operator != LayerOperatorRecurrentMix || state.kind != stateSpaceKeyedDelta ||
		feedForward.Operator != LayerOperatorFeedForwardStandardSwiGLU {
		t.Fatalf("Kimi recurrent policies = %+v/%+v/%d", mixer, feedForward, state.kind)
	}
}

func TestLFM2RecurrentProgramUsesSharedStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Recurrent: true},
		ArchitectureProfile{Attention: AttentionLFM2}, BlockDense,
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardStandardSwiGLU, LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	state := Spec{RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true}}}.
		withProfile(ArchitectureProfile{Attention: AttentionLFM2}).stateSpacePlan(0, true)
	if mixer.Operator != LayerOperatorRecurrentMix || state.kind != stateSpaceLFM2 ||
		feedForward.Operator != LayerOperatorFeedForwardStandardSwiGLU {
		t.Fatalf("LFM2 recurrent policies = %+v/%d/%d", mixer, state.kind, feedForward.Operator)
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
			program := compileLayerProgram(
				LayerPlan{
					Recurrent:   fixture.recurrent,
					Composition: fixture.composition,
				},
				ArchitectureProfile{}, BlockNemotronH,
			)
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
			ArchitectureProfile{Attention: AttentionQwenGDN}, BlockDense,
		)
		instruction, ok := program.Instruction(mixerStage)
		if !ok || program.Count != qwenProgramStageCount {
			t.Fatalf("Qwen GDN program = %+v", program)
		}
		state := Spec{RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{recurrent}}}.
			withProfile(ArchitectureProfile{Attention: AttentionQwenGDN}).stateSpacePlan(0, recurrent)
		if recurrent && (instruction.Operator != LayerOperatorRecurrentMix || state.kind != stateSpaceQwenGDN) {
			t.Fatalf("Qwen recurrent stage = %+v", instruction)
		}
		if !recurrent && instruction.Operator != LayerOperatorAttentionGatedProjection {
			t.Fatalf("Qwen attention stage = %+v", instruction)
		}
	}
}

func TestArchitectureBlockDispatchRoutesFamilies(t *testing.T) {
	_, err := executeCompiledLayer(BlockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: "unknown"}},
	})
	var unsupported *UnsupportedArchitectureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("unknown error = %v", err)
	}
	t5 := Spec{CommonSpec: CommonSpec{Architecture: "t5"}}
	t5Plan := t5.PlanLayer(0, false)
	_, err = executeCompiledLayer(BlockDispatchOptions{Spec: t5, Plan: &t5Plan})
	if err == nil || !strings.Contains(err.Error(), "explicit encoder state") {
		t.Fatalf("T5 dispatch error = %v", err)
	}
	external := ArchitectureProfile{
		Name: "external-encoder-decoder", GraphFamily: ArchitectureFamilyEncoderDecoder,
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
