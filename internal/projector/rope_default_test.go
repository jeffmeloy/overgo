package projector

import (
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

func ropeDefaultFixture(t *testing.T, extra ...gguf.Metadata) *gguf.File {
	t.Helper()
	scalar := func(key string, kind gguf.ValueType, data any) gguf.Metadata {
		return gguf.Metadata{Key: key, Value: gguf.Value{Type: kind, Data: data}}
	}
	array := func(key string, kind gguf.ValueType, data any) gguf.Metadata {
		return gguf.Metadata{Key: key, Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: kind, Data: data}}
	}
	metadata := []gguf.Metadata{
		scalar("general.architecture", gguf.ValueTypeString, "clip"),
		scalar(visionEncoderEnabledKey, gguf.ValueTypeBool, true),
		scalar(visionProjectorTypeKey, gguf.ValueTypeString, qwen3VLProjectorType),
		scalar(visionImageSizeKey, gguf.ValueTypeUint32, uint32(768)),
		scalar(visionPatchSizeKey, gguf.ValueTypeUint32, uint32(16)),
		scalar(visionHiddenKey, gguf.ValueTypeUint32, uint32(1152)),
		scalar(visionIntermediateKey, gguf.ValueTypeUint32, uint32(4304)),
		scalar(visionProjectionKey, gguf.ValueTypeUint32, uint32(5120)),
		scalar(visionLayerCountKey, gguf.ValueTypeUint32, uint32(27)),
		scalar(visionHeadCountKey, gguf.ValueTypeUint32, uint32(16)),
		scalar(visionNormEpsilonKey, gguf.ValueTypeFloat32, float32(1e-6)),
		array(visionImageMeanKey, gguf.ValueTypeFloat32, []float32{0.5, 0.5, 0.5}),
		array(visionImageStandardKey, gguf.ValueTypeFloat32, []float32{0.5, 0.5, 0.5}),
	}
	path := filepath.Join(t.TempDir(), "mmproj.gguf")
	tensors := []gguf.TensorData{testutil.GGUFTensorF32("v.patch_embd.weight", []uint64{4, 4}, 1)}
	if err := gguf.WriteFileExclusive(path, append(metadata, extra...), tensors, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = file.Close() })
	return file
}

// TestRotaryVisionBackboneDefaultsRopeFrequency pins the shared-reader
// tolerance contract: a converter that omits the vision rope base gets
// the reference runtime's theta 10000, a declared base is honored, and
// a present key of the wrong type still refuses.
func TestRotaryVisionBackboneDefaultsRopeFrequency(t *testing.T) {
	var spec visionBackboneSpec
	projection := 0
	if err := readRotaryVisionBackbone(ropeDefaultFixture(t), qwen3VLProjectorType, &projection, &spec); err != nil {
		t.Fatalf("absent rope base refused: %v", err)
	}
	if spec.RopeFrequency != defaultVisionRopeFrequency {
		t.Fatalf("absent rope base = %v, want the reference default %v", spec.RopeFrequency, float32(defaultVisionRopeFrequency))
	}

	declared := ropeDefaultFixture(t, gguf.Metadata{
		Key: visionRopeFrequencyKey, Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(1e6)},
	})
	if err := readRotaryVisionBackbone(declared, qwen3VLProjectorType, &projection, &spec); err != nil {
		t.Fatalf("declared rope base refused: %v", err)
	}
	if spec.RopeFrequency != 1e6 {
		t.Fatalf("declared rope base = %v, want the declaration", spec.RopeFrequency)
	}

	wrongType := ropeDefaultFixture(t, gguf.Metadata{
		Key: visionRopeFrequencyKey, Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(10000)},
	})
	if err := readRotaryVisionBackbone(wrongType, qwen3VLProjectorType, &projection, &spec); err == nil {
		t.Fatal("wrong-typed rope base accepted; a present declaration must be exact")
	}
}
