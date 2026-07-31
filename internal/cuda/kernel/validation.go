package kernel

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

const (
	// BundleABIVersion: incremented for incompatible kernel launch contracts
	BundleABIVersion = 8
	// BundleTarget: PTX virtual architecture pinned by manifest
	BundleTarget = "compute_89"

	VectorAddSHA256 = "af74febeae5bb087f35610f7d194695c10374bc7aed4a224da6072a56e59d159"
	OpsF32SHA256    = "18406b5ece8f27f003c1472ec208deb6f8268cbdb0f06aaed6b786954610c248"
)

// ValidateAssets: fails closed when embedded PTX differs from kernel bundle
// whose launch ABI was compiled into Go host; CUDA's own JIT cache is
// content-addressed; these pins ensure host never submits unrecognized
// module to it
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
