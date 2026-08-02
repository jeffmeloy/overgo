package kernel

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

const (
	// BundleABIVersion: incompatible launch-contract revision.
	BundleABIVersion = 26
	// BundleTarget: pinned PTX virtual architecture.
	BundleTarget = "compute_89"

	VectorAddSHA256 = "af74febeae5bb087f35610f7d194695c10374bc7aed4a224da6072a56e59d159"
	OpsF32SHA256    = "2bba7002dc5395c2d4bbc85d6a9fe31cbf1994b589c83a011a2e3ecfac577898"
)

// ValidateAssets: reject PTX outside pinned host ABI.
func ValidateAssets() error {
	if err := validateAsset("vector_add.ptx", VectorAddPTX, VectorAddSHA256); err != nil {
		return err
	}
	return validateAsset("ops_f32.ptx", OpsF32PTX, OpsF32SHA256)
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
