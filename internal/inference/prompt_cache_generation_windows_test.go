//go:build windows

package inference

import "testing"

func TestPromptCacheReuseAfterDecodeTokens(t *testing.T) {
	assertPromptCacheAfterDecode(t, openHermeticScoringRunner(t))
}
