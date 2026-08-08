package model

import (
	"errors"
	"strings"
	"testing"
)

func TestLayerProgramsCoverCompiledPolicies(t *testing.T) {
	for policy := BlockDense; policy <= BlockQwenGDN; policy++ {
		program := compileLayerProgram(policy, false)
		instruction, ok := program.Instruction(0)
		if !ok {
			t.Fatalf("block policy %d has no compiled operator", policy)
		}
		var want []LayerOperator
		switch policy {
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
		}
		if len(want) != 0 {
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
			continue
		}
		if program.Count != 1 || instruction.Operator != LayerOperatorFamilyBlock ||
			instruction.Family != policy {
			t.Fatalf("block policy %d family program = %+v", policy, program)
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
