package runrecord

import (
	"slices"
	"strings"
	"testing"
)

// TestParseVerify holds the one verifier parser to what the shell runs:
// quotes group words and hide operators, && chains plain segments, env
// assignments and file guards are recognized, flags with separate values
// consume them, a wrapper's forwarded -run is its target, and anything
// else the shell composes marks the command composite instead of failing.
func TestParseVerify(t *testing.T) {
	t.Parallel()
	command := `test -f internal/x/a_test.go && OVERGO_CUDA_TEST=1 go test -timeout 20m ./internal/x ./internal/y -run '^(TestA|TestB)$' -count=1 -short && go run ./cmd/webui-lane -run "^TestWebUI$" -require 'a && b'`
	parsed, err := ParseVerify(command)
	if err != nil || parsed.Composite || len(parsed.Segments) != 3 {
		t.Fatalf("parse = %+v, %v", parsed, err)
	}
	guard, test, wrapper := parsed.Segments[0], parsed.Segments[1], parsed.Segments[2]
	if guard.Kind != SegmentFileGuard || guard.Path != "internal/x/a_test.go" {
		t.Fatalf("guard = %+v", guard)
	}
	if test.Kind != SegmentGoTest || !slices.Equal(test.Env, []string{"OVERGO_CUDA_TEST=1"}) ||
		!slices.Equal(test.Packages, []string{"./internal/x", "./internal/y"}) || !test.Short ||
		!slices.Equal(test.Flags, []string{"-timeout=20m", "-count=1"}) || test.Target == nil || !test.Target.MatchString("TestB") {
		t.Fatalf("go test = %+v", test)
	}
	if wrapper.Kind != 0 || wrapper.Target == nil || wrapper.Target.Pattern != "^TestWebUI$" || wrapper.Fields[len(wrapper.Fields)-1] != "a && b" {
		t.Fatalf("wrapper = %+v", wrapper)
	}
	if targets := parsed.Targets(); len(targets) != 2 || !parsed.Short() {
		t.Fatalf("targets = %d short = %v", len(targets), parsed.Short())
	}
	json := parsed.JSONCommand(command)
	if !strings.Contains(json, "OVERGO_CUDA_TEST=1 go test -json -timeout 20m") || strings.Contains(json, "go run -json") {
		t.Fatalf("json command = %s", json)
	}
	for _, composite := range []string{"go run ./cmd/x > /dev/null && go test ./a -count=1", "go test ./a | tee log", "test $(ls | wc -l) -eq 0", `test -n "$HOME" && go test ./a`} {
		parsed, err := ParseVerify(composite)
		if err != nil || !parsed.Composite {
			t.Fatalf("%q: composite=%v err=%v", composite, parsed.Composite, err)
		}
	}
	for _, refused := range []string{"go test ./a -run 'open", "go test ./a -run", "go test ./a -run A -run B", `go test ./a \`} {
		if _, err := ParseVerify(refused); err == nil {
			t.Fatalf("%q parsed", refused)
		}
	}
}
