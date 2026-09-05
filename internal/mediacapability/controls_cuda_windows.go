//go:build windows

package mediacapability

import (
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/sensenovarecipe"
)

// The device-bound image requests exist only where their runtimes build.
func init() {
	requestTypes[modelrecipe.ModuleLatentImagePrepare] = latentimage.Request{}
	requestTypes[modelrecipe.ModuleRoutedImagePrepare] = sensenovarecipe.GenerationRequest{}
}
