package thoughtbank

import "testing"

func TestGenerateRequestHasNoPackageLocalTokenCeiling(t *testing.T) {
	request := GenerateRequest{Text: "continue", MaxTokens: 257}
	if err := ValidateGenerateRequest(request); err != nil {
		t.Fatalf("positive request was rejected by a package-local token ceiling: %v", err)
	}
}
