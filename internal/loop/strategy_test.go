package loop

import (
	"strings"
	"testing"
)

// TestStrategyIdentity pins the identity contract: a declared name
// wins, an undeclared configuration derives a stable digest of its
// exact command line, and different commands never share a derived
// identity.
func TestStrategyIdentity(t *testing.T) {
	if got := StrategyIdentity([]string{"claude", "-p", "{prompt}"}, "sonnet-baseline"); got != "sonnet-baseline" {
		t.Fatalf("declared name = %q", got)
	}
	first := StrategyIdentity([]string{"claude", "-p", "{prompt}"}, "")
	again := StrategyIdentity([]string{"claude", "-p", "{prompt}"}, "")
	other := StrategyIdentity([]string{"gpt", "-p", "{prompt}"}, "")
	if first == "" || first != again {
		t.Fatalf("derived identity is not stable: %q vs %q", first, again)
	}
	if !strings.HasPrefix(first, "worker-") || first == other {
		t.Fatalf("derived identities = %q, %q", first, other)
	}
	// Argument boundaries matter: ["ab","c"] and ["a","bc"] are
	// different commands and must not collide through joining.
	if StrategyIdentity([]string{"ab", "c"}, "") == StrategyIdentity([]string{"a", "bc"}, "") {
		t.Fatal("argument boundaries collapsed in the digest")
	}
	if got := StrategyIdentity(nil, ""); got != "" {
		t.Fatalf("empty worker = %q, want undeclared", got)
	}
}
