package thoughtbank

import (
	"testing"

	"overgo/internal/textgeneration"
)

func TestGenerateRequestHasNoPackageLocalTokenCeiling(t *testing.T) {
	request := textgeneration.Request{Text: "continue", MaxTokens: 257}
	if err := textgeneration.Validate(request); err != nil {
		t.Fatalf("positive request was rejected by a package-local token ceiling: %v", err)
	}
}
