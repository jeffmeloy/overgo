package kernel

import (
	"fmt"
	"strings"
	"testing"
)

func TestValidateAssets(t *testing.T) {
	if err := ValidateAssets(); err != nil {
		t.Fatal(err)
	}
}

func TestValidateAssetRejectsDifferentContent(t *testing.T) {
	err := validateAsset("changed.ptx", []byte("changed"), strings.Repeat("0", 64))
	if err == nil {
		t.Fatal("changed kernel asset was accepted")
	}
	if !strings.Contains(err.Error(), fmt.Sprintf("ABI v%d", BundleABIVersion)) ||
		!strings.Contains(err.Error(), BundleTarget) {
		t.Fatalf("error lacks bundle identity: %v", err)
	}
}
