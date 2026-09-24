package gate

import (
	"slices"
	"testing"

	"overgo/internal/runrecord"
)

// TestVerifierParsedOnce holds the gate's readers of a verify command to the
// one parse: the checkpoint memo takes its packages and guards from it, so
// a quoted && inside a selector is not a segment boundary and a package
// named after a flag is still an input; and the memo refuses what its input
// could not describe, an environment prefix or a composed command. The
// evidence targets come from the same parse, so both see the same selector.
func TestVerifierParsedOnce(t *testing.T) {
	t.Parallel()
	verify := `test -f internal/x/a_test.go && go test -count=1 ./internal/x -run '^(TestA|TestB)$' && go test ./internal/y -run 'a && b' -count=1`
	packages, guards, err := verifierInputs(verify)
	if err != nil || !slices.Equal(packages, []string{"./internal/x", "./internal/y"}) || !slices.Equal(guards, []string{"internal/x/a_test.go"}) {
		t.Fatalf("memo inputs = %v %v %v", packages, guards, err)
	}
	targets, err := runrecord.GoTestTargets(verify, true)
	if err != nil || len(targets) != 2 || targets[1].Pattern != "a && b" {
		t.Fatalf("evidence targets = %v, %v", targets, err)
	}
	for _, uncacheable := range []string{
		"OVERGO_CUDA_TEST=1 go test ./internal/x -run '^TestA$' -count=1",
		"go run ./cmd/x > /dev/null && go test ./internal/x -run '^TestA$' -count=1",
		"go test ./internal/x -run '^TestA$' -count=1 && go run ./cmd/x",
	} {
		if _, _, err := verifierInputs(uncacheable); err == nil {
			t.Fatalf("%q was memoisable", uncacheable)
		}
	}
}
