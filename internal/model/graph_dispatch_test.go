package model

import (
	"errors"
	"strings"
	"testing"
)

func TestArchitectureBlockDispatchRoutesFamilies(t *testing.T) {
	_, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: "unknown"}},
	})
	var unsupported *UnsupportedArchitectureError
	if !errors.As(err, &unsupported) {
		t.Fatalf("unknown error = %v", err)
	}
	_, err = BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: "t5"}},
	})
	if err == nil || !strings.Contains(err.Error(), "explicit encoder state") {
		t.Fatalf("T5 dispatch error = %v", err)
	}
	external := ArchitectureProfile{
		Name: "external-encoder-decoder", GraphFamily: ArchitectureFamilyEncoderDecoder,
	}
	_, err = BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: Spec{CommonSpec: CommonSpec{Architecture: external.Name}}.withProfile(external),
	})
	if err == nil || !strings.Contains(err.Error(), "explicit encoder state") {
		t.Fatalf("bound external dispatch error = %v", err)
	}
}
