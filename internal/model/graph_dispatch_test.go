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
		program := compileLayerProgram(policy, AttentionStandard, false, composition)
		_, ok := program.Instruction(0)
		if !ok {
			t.Fatalf("block policy %d has no compiled operator", policy)
		}
		var want []LayerOperator
		switch policy {
		case BlockDense:
			want = []LayerOperator{LayerOperatorDenseTransformer}
		case BlockKimiLinear:
			want = []LayerOperator{LayerOperatorLatentAttention}
		case BlockMLA:
			want = []LayerOperator{LayerOperatorLatentAttention}
		case BlockDSA:
			want = []LayerOperator{LayerOperatorLatentAttention}
		case BlockDeepSeek4:
			want = []LayerOperator{LayerOperatorHyperConnection}
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

func TestKimiRecurrentProgramSelectsLinearAttention(t *testing.T) {
	program := compileLayerProgram(
		BlockKimiLinear, AttentionStandard, true, LayerCompositionStandard,
	)
	instruction, ok := program.Instruction(0)
	if !ok || program.Count != 1 || instruction.Operator != LayerOperatorLinearAttention {
		t.Fatalf("Kimi recurrent program = %+v", program)
	}
}

func TestLayerDispatchRejectsMutatedProgram(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "llama", BlockCount: 1}}
	plan := spec.PlanLayer(0, false)
	plan.Program.Instructions[0].Operator = LayerOperatorNone
	_, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: spec, Plan: &plan, Context: CachedBlockContext{},
	})
	if err == nil || !strings.Contains(err.Error(), "differs from layer policy") {
		t.Fatalf("mutated program error = %v", err)
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
				BlockNemotronH, AttentionStandard, fixture.recurrent, fixture.composition,
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
			BlockDense, AttentionQwenGDN, recurrent, LayerCompositionStandard,
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
