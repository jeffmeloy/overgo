//go:build windows

package inference

import "testing"

func TestPromptCacheReuseAfterDecodeTokens(t *testing.T) {
	assertPromptCacheAfterDecode(t, openHermeticScoringRunner(t))
}

func TestPromptCacheRetainsDecodedTokensDevice(t *testing.T) {
	assertDecodedTokensContinue(t, openHermeticScoringRunner(t))
}
