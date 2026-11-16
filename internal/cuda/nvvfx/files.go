package nvvfx

import (
	"fmt"
	"slices"

	"overgo/internal/media"
)

// runtimeFiles are the runtime's libraries in the order NVIDIA's binding loads them (nvidia-vfx 0.1.0.1, _lib_loader.py):
// each part before the libraries that ask for it by name, so a name asked for is a library already loaded.
var runtimeFiles = []string{
	"nppc64_12.dll", "nppial64_12.dll", "nppicc64_12.dll", "nppidei64_12.dll", "nppif64_12.dll", "nppig64_12.dll",
	"nppim64_12.dll", "nppist64_12.dll", "nppitc64_12.dll", "nvinfer_10.dll", "nvinfer_plugin_10.dll",
	"nvonnxparser_10.dll", imageLibrary, "nvngxruntime.dll", effectsLibrary, "nvVFXVideoSuperRes.dll", "nvngx_vsr.dll",
}

// The two libraries whose entry points overgo calls.
const (
	imageLibrary   = "NVCVImage.dll"
	effectsLibrary = "NVVideoEffects.dll"
)

// Found is the directory the runtime is loaded from, the one the data root's record or NVIDIA's installer names, that
// holds every library of it, or why there is none (media.EffectsRuntime).
func Found() (string, error) {
	directory, err := media.EffectsRuntime(runtimeFiles)
	if err != nil {
		return "", fmt.Errorf("nvvfx: %w", err)
	}
	return directory, nil
}

// Files are the runtime's libraries, which a directory recorded as the runtime's holds.
func Files() []string { return slices.Clone(runtimeFiles) }
