package latentimage

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// scriptedChat answers each request with the next of its answers, keeping the system prompts, requests and lengths
// it was asked with.
type scriptedChat struct {
	answers           []string
	systems, requests []string
	lengths           []int
}

func (s *scriptedChat) chat(_ context.Context, system, request string, maxTokens int) (string, error) {
	if len(s.answers) == 0 {
		return "", errors.New("the script has no answer left")
	}
	s.systems, s.requests, s.lengths = append(s.systems, system), append(s.requests, request), append(s.lengths, maxTokens)
	answer := s.answers[0]
	s.answers = s.answers[1:]
	return answer, nil
}

// TestModeTwoRunsTheEnhancersChain holds prompt mode 2's chain (moss-ultra-sections/prompt-modes) about a scripted
// chat: a first answer of enough words is the positive and asks no retry; the negative is asked of that positive under
// the photograph's system prompt for a photographic source, at the negative's length, and cleaned; a first answer too
// short asks the retry with it, the source standing where both are empty; a source naming no photograph is asked under
// the protected system prompt, and a negative of none gives the user's exclusions.
func TestModeTwoRunsTheEnhancersChain(t *testing.T) {
	data := testcheck.Must(mossEnhancer())(t)
	long := strings.Repeat("word ", enhancerWords)
	photo := &scriptedChat{answers: []string{long, "photo, blurry, blurry"}}
	positive, negative := enhanced(t, photo, "a photograph of a lighthouse", "")
	if positive != long || negative != "blurry" || len(photo.requests) != 2 {
		t.Fatalf("a long first answer gave %q and %q in %d requests; want it, blurry, and two requests", positive[:min(len(positive), 20)], negative, len(photo.requests))
	}
	if photo.systems[0] != data.PositiveSystem || photo.systems[1] != data.NegativeSystemPhoto ||
		photo.requests[1] != enhancerFormat(data.NegativeRequest, "a photograph of a lighthouse", long, "") ||
		photo.lengths[0] != enhancerPositiveTokens || photo.lengths[1] != enhancerNegativeTokens {
		t.Fatalf("the photograph's chain asked %v at %v", photo.systems, photo.lengths)
	}

	short := &scriptedChat{answers: []string{"too short", "", "None."}}
	positive, negative = enhanced(t, short, "an anime cat", "no dogs")
	if positive != "an anime cat" || negative != "no dogs" || len(short.requests) != 3 {
		t.Fatalf("an empty retry gave %q and %q in %d requests; want the source, the exclusions, and three requests", positive, negative, len(short.requests))
	}
	if short.requests[1] != enhancerFormat(data.RetryRequest, "an anime cat", "no dogs", "too short") || short.systems[2] != data.NegativeSystemProtected {
		t.Fatalf("the short answer's retry asked %q under %q", short.requests[1], short.systems[2])
	}
	if _, _, err := enhance(t.Context(), (&scriptedChat{}).chat, "a cat", ""); err == nil {
		t.Fatal("a chat that failed gave an enhanced prompt")
	}
}

// enhanced is mode 2's positive and negative about the scripted chat, failing the test where the chain errs.
func enhanced(t *testing.T, chat *scriptedChat, source, exclusions string) (string, string) {
	t.Helper()
	positive, negative, err := enhance(t.Context(), chat.chat, source, exclusions)
	testcheck.NoError(t, err)
	return positive, negative
}
