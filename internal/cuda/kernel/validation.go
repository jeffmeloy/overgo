package kernel

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

const (
	// BundleABIVersion is incremented for incompatible kernel launch contracts.
	BundleABIVersion = 4
	// BundleTarget is the PTX virtual architecture pinned by the manifest.
	BundleTarget = "compute_89"

	VectorAddSHA256 = "af74febeae5bb087f35610f7d194695c10374bc7aed4a224da6072a56e59d159"
	OpsF32SHA256    = "3eec568828012091e6d116c15d407a1fe68f498a287083990f010f9b03a88042"
)

// ValidateAssets fails closed when embedded PTX differs from the kernel bundle
// whose launch ABI was compiled into the Go host. CUDA's own JIT cache is
// content-addressed; these pins ensure the host never submits an unrecognized
// module to it.
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
