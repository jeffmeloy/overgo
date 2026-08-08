package model

import (
	"errors"
	"strings"
	"testing"
)

func TestLayerProgramsCoverCompiledPolicies(t *testing.T) {
	for policy := BlockDense; policy <= BlockDeepSeek4; policy++ {
		composition := LayerCompositionStandard
		if policy == BlockNemotronH {
			composition = LayerCompositionAttentionOnly
		}
		program := compileLayerProgram(
			LayerPlan{Block: policy, Composition: composition}, ArchitectureProfile{},
		)
		_, ok := program.Instruction(0)
		if !ok {
			t.Fatalf("block policy %d has no compiled operator", policy)
		}
		var want []LayerOperator
		switch policy {
		case BlockDense:
			want = []LayerOperator{
				LayerOperatorDenseAttention, LayerOperatorDenseFeedForward,
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
		if program.Count != uint8(len(want)) {
			t.Fatalf("semantic program %d count = %d", policy, program.Count)
		}
		for index, operator := range want {
			instruction, _ := program.Instruction(index)
			if instruction.Operator != operator {
				t.Fatalf("semantic program %d stage %d = %d, want %d", policy, index, instruction.Operator, operator)
			}
			if operator == LayerOperatorRecurrentMix && instruction.Recurrent == RecurrentMixNone {
				t.Fatalf("semantic program %d recurrent stage has no math policy", policy)
			}
		}
	}
}

func TestDenseProgramsIsolateAtomicGraphs(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockDense}, ArchitectureProfile{DenseGraph: DenseGraphBERT},
	)
	instruction, ok := program.Instruction(0)
	if !ok || program.Count != 1 || instruction.Operator != LayerOperatorDenseTransformer {
		t.Fatalf("atomic dense program = %+v", program)
	}
}

func TestDeciSparseProgramUsesNeutralStages(t *testing.T) {
	program := compileLayerProgram(LayerPlan{Block: BlockDense, DeciSparse: true}, ArchitectureProfile{})
	want := []LayerOperator{
		LayerOperatorCacheSentinel, LayerOperatorAttentionNorm, LayerOperatorAttentionMix,
		LayerOperatorResidual, LayerOperatorFeedForwardNorm, LayerOperatorFeedForwardMix,
		LayerOperatorResidual,
	}
	if program.Count != uint8(len(want)) {
		t.Fatalf("Deci stage count = %d", program.Count)
	}
	for index, operator := range want {
		instruction, _ := program.Instruction(index)
		if instruction.Operator != operator {
			t.Fatalf("Deci stage %d = %d, want %d", index, instruction.Operator, operator)
		}
	}
}

func TestKimiRecurrentProgramSelectsLinearAttention(t *testing.T) {
	program := compileLayerProgram(
		LayerPlan{Block: BlockKimiLinear, Recurrent: true}, ArchitectureProfile{},
	)
	instruction, ok := program.Instruction(0)
	if !ok || program.Count != 6 || instruction.Operator != LayerOperatorAttentionNorm {
		t.Fatalf("Kimi recurrent program = %+v", program)
	}
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	if mixer.Operator != LayerOperatorLinearAttention ||
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
	if program.Count != uint8(len(want)) {
		t.Fatalf("LFM2 recurrent stage count = %d", program.Count)
	}
	for index, operator := range want {
		instruction, _ := program.Instruction(index)
		if instruction.Operator != operator {
			t.Fatalf("LFM2 recurrent stage %d = %d", index, instruction.Operator)
		}
	}
	mixer, _ := program.Instruction(1)
	feedForward, _ := program.Instruction(4)
	if mixer.Recurrent != RecurrentMixLFM2 || feedForward.FeedForward != FeedForwardMixStandardSwiGLU {
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
		if recurrent && instruction.Recurrent != RecurrentMixQwenGDN {
			t.Fatalf("Qwen recurrent stage = %+v", instruction)
		}
		if !recurrent && instruction.Attention != AttentionMixQwenGDN {
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
