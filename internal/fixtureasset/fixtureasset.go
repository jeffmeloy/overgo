// Package fixtureasset loads sha256-pinned raw little-endian f32 tensor
// assets — the single loader every golden-fixture consumer shares.
package fixtureasset

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// LoadF32 reads dir/asset, verifies its sha256 against the golden pin, and
// decodes little-endian f32 values. elements > 0 additionally requires that
// exact element count.
func LoadF32(dir, asset, sha string, elements int) ([]float32, error) {
	if asset == "" {
		return nil, fmt.Errorf("fixture asset: missing asset name")
	}
	if sha == "" {
		return nil, fmt.Errorf("fixture asset: %s: golden missing sha256", asset)
	}
	raw, err := os.ReadFile(filepath.Join(dir, asset))
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); !strings.EqualFold(got, sha) {
		return nil, fmt.Errorf("fixture asset: %s sha256 %s != golden %s", asset, got, sha)
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("fixture asset: %s: %d bytes not float32-aligned", asset, len(raw))
	}
	values := make([]float32, len(raw)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	if elements > 0 && len(values) != elements {
		return nil, fmt.Errorf("fixture asset: %s has %d values, want %d", asset, len(values), elements)
	}
	return values, nil
}
