package model

import (
	"errors"
	"strings"
	"testing"
)

func TestCachedBlockCatalogCoversCompiledPolicies(t *testing.T) {
	for policy := BlockDense; policy <= BlockQwenGDN; policy++ {
		if int(policy) >= len(cachedBlockCatalog) || cachedBlockCatalog[policy] == nil {
			t.Fatalf("block policy %d has no cached builder", policy)
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
