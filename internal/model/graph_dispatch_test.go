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
		switch operator {
		case LayerOperatorRecurrentMix:
			if instruction.Recurrent == RecurrentMixNone {
				t.Fatalf("stage %d has no recurrent policy", index)
			}
		case LayerOperatorAttentionMix:
			if instruction.Attention == AttentionMixNone {
				t.Fatalf("stage %d has no attention policy", index)
			}
		case LayerOperatorFeedForwardMix:
			if instruction.FeedForward == FeedForwardMixNone {
				t.Fatalf("stage %d has no feed-forward policy", index)
			}
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
			LayerPlan{Block: policy, Composition: composition}, ArchitectureProfile{},
		)
		var want []LayerOperator
		switch policy {
		case BlockDense:
			want = []LayerOperator{
				LayerOperatorAttentionInputNorm, LayerOperatorAttentionMix, LayerOperatorResidual,
				LayerOperatorFeedForwardInputNorm, LayerOperatorFeedForwardMix,
				LayerOperatorFeedForwardOutput, LayerOperatorResidual,
			}
		case BlockKimiLinear:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorLatentAttention, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
			}
		case BlockMLA:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorLatentAttention, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
			}
		case BlockDSA:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorLatentAttention, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
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
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
			}
		case BlockGraniteHybrid:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorScale,
				LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix,
				LayerOperatorScale, LayerOperatorResidual,
			}
		case BlockPLaMo2:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorAttentionPostNorm,
				LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix,
				LayerOperatorFeedForwardPostNorm, LayerOperatorResidual,
			}
		case BlockNemotronH:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorAttentionMix, LayerOperatorResidual,
			}
		case BlockFalconH1:
			want = []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorHybridMix, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
			}
		}
		requireLayerProgram(t, program, want...)
	}
}

func TestGemma4ProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockDense}, ArchitectureProfile{DenseGraph: DenseGraphGemma4},
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorAttentionMix, LayerOperatorAttentionPostNorm,
		LayerOperatorResidual, LayerOperatorFeedForwardMix, LayerOperatorResidual,
		LayerOperatorOutputAdapter,
	}
	requireLayerProgram(t, program, want...)
}

func TestEagle3ProgramUsesPairedInputStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockDense}, ArchitectureProfile{Forward: ForwardEagle3},
	)
	want := []LayerOperator{
		LayerOperatorPairedInputNorm, LayerOperatorAttentionMix, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
	paired, _ := program.Instruction(0)
	if paired.TensorCount != 1 || paired.Tensors[0] != RuntimeTensorPerLayerInput {
		t.Fatalf("Eagle3 paired binding = %+v", paired)
	}
}

func TestGemma4AssistantProgramUsesSharedCacheStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockDense}, ArchitectureProfile{Forward: ForwardGemma4Assistant},
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorAttentionMix, LayerOperatorAttentionPostNorm,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix,
		LayerOperatorFeedForwardPostNorm, LayerOperatorResidualScale,
	}
	requireLayerProgram(t, program, want...)
	attention, _ := program.Instruction(1)
	if attention.Attention != AttentionMixSharedCacheQKNorm || attention.CacheCount != 2 {
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
				LayerOperatorTokenShiftMix, LayerOperatorResidual, LayerOperatorPeriodicScale,
			},
		},
		{
			name: "RWKV7 channel", profile: ArchitectureProfile{
				DenseGraph: DenseGraphRWKV7, Normalization: NormalizationLayer,
			},
			want: []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
				LayerOperatorTokenShiftMix, LayerOperatorResidual,
			},
		},
		{
			name: "RWKV7 SwiGLU", profile: ArchitectureProfile{
				DenseGraph: DenseGraphRWKV7, Normalization: NormalizationRMS,
			},
			want: []LayerOperator{
				LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
				LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			program := compileLayerProgram(LayerPlan{Block: BlockDense}, test.profile)
			requireLayerProgram(t, program, test.want...)
		})
	}
}

func TestTalkieProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockDense}, ArchitectureProfile{DenseGraph: DenseGraphTalkie},
	)
	want := []LayerOperator{
		LayerOperatorRMSNorm, LayerOperatorAttentionMix, LayerOperatorResidual,
		LayerOperatorRMSNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
		LayerOperatorScaledSkip,
	}
	requireLayerProgram(t, program, want...)
}

func TestBERTProgramUsesPostNormalizedStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockDense}, ArchitectureProfile{DenseGraph: DenseGraphBERT},
	)
	want := []LayerOperator{
		LayerOperatorAttentionMix, LayerOperatorAttentionResidualNorm,
		LayerOperatorFeedForwardMix, LayerOperatorFeedForwardResidualNorm,
	}
	requireLayerProgram(t, program, want...)
}

func TestJinaV2ProgramAddsInputResidualNormalization(t *testing.T) {
	program := compileLayerProgram(LayerPlan{Block: BlockDense}, ArchitectureProfile{
		DenseGraph: DenseGraphBERT, EncoderGraph: EncoderGraphPolicy{Kind: encoderGraphJinaV2},
	})
	instruction, ok := program.Instruction(2)
	if !ok || program.Count != 5 || instruction.Operator != LayerOperatorInputResidualNorm {
		t.Fatalf("JinaV2 program = %+v", program)
	}
}

func TestGemmaEmbeddingProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockDense}, ArchitectureProfile{DenseGraph: DenseGraphGemmaEmbedding},
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorAttentionMix, LayerOperatorAttentionPostNorm,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix,
		LayerOperatorFeedForwardPostNorm, LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
}

func TestDeciSparseProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(LayerPlan{Block: BlockDense, DeciSparse: true}, ArchitectureProfile{})
	want := []LayerOperator{
		LayerOperatorCacheSentinel, LayerOperatorAttentionNorm, LayerOperatorAttentionMix,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix,
		LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
}

func TestKimiRecurrentProgramSelectsLinearAttention(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockKimiLinear, Recurrent: true}, ArchitectureProfile{},
	)
	requireLayerProgram(t, program,
		LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
	)
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	if mixer.Operator != LayerOperatorRecurrentMix || mixer.Recurrent != RecurrentMixKeyedDeltaAttention ||
		feedForward.FeedForward != FeedForwardMixStandardSwiGLU {
		t.Fatalf("Kimi recurrent policies = %+v/%+v", mixer, feedForward)
	}
}

func TestLFM2RecurrentProgramUsesSharedStages(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockDense, Recurrent: true},
		ArchitectureProfile{Attention: AttentionLFM2},
	)
	want := []LayerOperator{
		LayerOperatorAttentionNorm, LayerOperatorRecurrentMix, LayerOperatorResidual,
		LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix, LayerOperatorResidual,
	}
	requireLayerProgram(t, program, want...)
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	if mixer.Recurrent != RecurrentMixShortConvolution || feedForward.FeedForward != FeedForwardMixStandardSwiGLU {
		t.Fatalf("LFM2 recurrent policies = %d/%d", mixer.Recurrent, feedForward.FeedForward)
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
		{name: "attention", composition: LayerCompositionAttentionOnly, mixer: LayerOperatorAttentionMix},
		{name: "feed-forward", composition: LayerCompositionFeedForwardOnly, mixer: LayerOperatorCacheSentinel},
	}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			program := compileLayerProgram(
				LayerPlan{
					Block: BlockNemotronH, Recurrent: fixture.recurrent,
					Composition: fixture.composition,
				},
				ArchitectureProfile{},
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
			LayerPlan{Block: BlockDense, Recurrent: recurrent},
			ArchitectureProfile{Attention: AttentionQwenGDN},
		)
		instruction, ok := program.Instruction(mixerStage)
		if !ok || program.Count != qwenProgramStageCount {
			t.Fatalf("Qwen GDN program = %+v", program)
		}
		if recurrent && instruction.Recurrent != RecurrentMixGatedDelta {
			t.Fatalf("Qwen recurrent stage = %+v", instruction)
		}
		if !recurrent && instruction.Attention != AttentionMixGatedProjection {
			t.Fatalf("Qwen attention stage = %+v", instruction)
		}
	}
}

func TestArchitectureBlockDispatchRoutesFamilies(t *testing.T) {
	_, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: "unknown"}},
	})
	var unsupported *UnsupportedArchitectureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("unknown error = %v", err)
	}
	t5 := Spec{CommonSpec: CommonSpec{Architecture: "t5"}}
	t5Plan := t5.PlanLayer(0, false)
	_, err = BuildArchitectureBlockCached(BlockDispatchOptions{Spec: t5, Plan: &t5Plan})
	if err == nil || !strings.Contains(err.Error(), "explicit encoder state") {
		t.Fatalf("T5 dispatch error = %v", err)
	}
	external := ArchitectureProfile{
		Name: "external-encoder-decoder", GraphFamily: ArchitectureFamilyEncoderDecoder,
	}
	externalSpec := Spec{CommonSpec: CommonSpec{Architecture: external.Name}}.withProfile(external)
	externalPlan := externalSpec.PlanLayer(0, false)
	_, err = BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: externalSpec, Plan: &externalPlan,
	})
	if err == nil || !strings.Contains(err.Error(), "explicit encoder state") {
		t.Fatalf("bound external dispatch error = %v", err)
	}
	llama, _ := LookupArchitecture("llama")
	_, err = BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: llama.Name}}.withProfile(llama),
	})
	if err == nil || !strings.Contains(err.Error(), "compiled layer plan is required") {
		t.Fatalf("missing plan error = %v", err)
	}
}
