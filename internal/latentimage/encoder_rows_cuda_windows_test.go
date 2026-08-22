//go:build windows

package latentimage

import (
	"path/filepath"

	"overgo/internal/safetensors"
)

func readEmbedRowsF32(modelDir string, spec TextEncoderSpec, ids []int) ([]float32, error) {
	return safetensors.ReadTensorRowsF32(
		filepath.Join(modelDir, "text_encoder"), textEncoderPrefix+"embed_tokens.weight", spec.Hidden, ids,
	)
}
