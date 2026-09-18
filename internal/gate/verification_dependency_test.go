package gate

import (
	"slices"
	"testing"
)

func TestVerificationDeclarationBoundary(t *testing.T) {
	t.Parallel()
	liveRepositoryFixture(t).use(t, func(g *gateContext) {
		const audio = "overgo/internal/audioparity"
		const evidence = "overgo/internal/testevidence"
		for _, mutation := range []struct {
			path  string
			audio bool
		}{
			{"internal/testevidence/evidence.go", false},
			{"internal/testevidence/invocations.go", false},
			{"internal/testevidence/package_updates.go", false},
			{"internal/testevidence/verdict.go", false},
			{"internal/runrecord/verify_policy.go", true},
			{"internal/runrecord/verify_selector.go", true},
			{"internal/evaluation/transcription_resources.go", true},
		} {
			t.Run(mutation.path, func(t *testing.T) {
				g.paths = []string{mutation.path}
				attribution, err := g.packageGraph.attributeInputs(audio, g.paths)
				if err != nil {
					t.Fatal(err)
				}
				if slices.Contains(attribution.CompilerInputs, mutation.path) != mutation.audio ||
					slices.Contains(attribution.UnboundInputs, mutation.path) == mutation.audio {
					t.Fatalf("audio input boundary: %+v", attribution)
				}
				scope, err := g.deriveTestScope()
				if err != nil {
					t.Fatal(err)
				}
				if slices.Contains(scope.selected(), audio) != mutation.audio {
					t.Fatalf("audio selected=%t, want %t; direct=%v dependent=%v uncertain=%v", slices.Contains(scope.selected(), audio), mutation.audio, scope.direct, scope.dependent, scope.uncertain)
				}
				if !mutation.audio && !slices.Contains(scope.direct, evidence) {
					t.Fatal("parser mutation lost its owning checks")
				}
				t.Logf("changed=%s audio_selected=%t selected=%d; no model acquisition", mutation.path, mutation.audio, len(scope.selected()))
			})
		}
	})
}
