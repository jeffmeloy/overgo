package main

import (
	"errors"
	"strings"
	"testing"
)

// TestHookRefusesStaleBuild holds the Stop hook to naming its own rebuild when
// a build older than the tree's sources cannot read the plan -- a stored
// schema change once blocked every turn end on a decode failure -- and to
// reporting the plan's own error when the build is current or unstamped.
func TestHookRefusesStaleBuild(t *testing.T) {
	t.Parallel()
	decode := errors.New(`strict decode: unknown field "class"`)
	notice := staleBuildNotice("0123456789abcdef", true, decode)
	for _, part := range []string{"go build -o bin/loophook.exe ./cmd/loophook", "0123456789ab", `unknown field "class"`} {
		if !strings.Contains(notice, part) {
			t.Errorf("stale-build notice lacks %q: %s", part, notice)
		}
	}
	if current := staleBuildNotice("0123456789abcdef", false, decode); current != "" {
		t.Errorf("a current build blamed itself: %s", current)
	}
	if unstamped := staleBuildNotice("", true, decode); unstamped != "" {
		t.Errorf("an unstamped build blamed itself: %s", unstamped)
	}
}
