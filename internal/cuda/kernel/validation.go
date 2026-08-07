package kernel

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

type bundleAsset struct {
	name   string
	data   []byte
	sha256 string
}

// ValidateAssets: reject PTX outside pinned host ABI.
func ValidateAssets() error {
	for _, asset := range bundleAssets {
		if err := validateAsset(asset.name, asset.data, asset.sha256); err != nil {
			return err
		}
	}
	return nil
}

func validateAsset(name string, data []byte, want string) error {
	sum := sha256.Sum256(data)
	got := hex.EncodeToString(sum[:])
	if got != want {
		return fmt.Errorf(
			"CUDA kernel bundle ABI v%d (%s): %s hash %s, want %s",
			BundleABIVersion,
			BundleTarget,
			name,
			got,
			want,
		)
	}
	return nil
}
