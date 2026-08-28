package media

import (
	"slices"
	"testing"
)

func TestExecuteCodecProgram(t *testing.T) {
	project, err := BindCodecWeights(CodecPointwise, false, []struct{}{{}, {}})
	if err != nil {
		t.Fatal(err)
	}
	head, err := BindCodecWeights(CodecHead, false, []struct{}{{}, {}, {}})
	if err != nil {
		t.Fatal(err)
	}
	projectOp := NewCodecOperation[struct{}](CodecPointwise, "project", 2, 3)
	projectOp.Bindings = project
	headOp := NewCodecOperation[struct{}](CodecHead, "head", 3, 1)
	headOp.Bindings = head
	program := CodecProgram[struct{}]{Operations: []CodecOperation[struct{}]{projectOp, headOp}}
	var visited []string
	output, err := ExecuteCodecProgram("test", program, make([]int, len(program.Operations)), CodecVolume[string]{
		Storage: "storage", Channels: 2, Frames: 1, Height: 4, Width: 4,
	}, false, func(_ int, operation CodecOperation[struct{}], state *int, current CodecVolume[string]) (CodecVolume[string], error) {
		*state++
		visited = append(visited, operation.Name)
		current.Channels = operation.OutputChannels
		return current, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if output.Channels != 1 || !slices.Equal(visited, program.Names()) {
		t.Fatalf("output=%+v visited=%v program=%v", output, visited, program.Names())
	}
	if _, err := ExecuteCodecProgram("mismatch", program, make([]int, 1), CodecVolume[int]{Channels: 2, Frames: 1, Height: 1, Width: 1}, false, func(_ int, operation CodecOperation[struct{}], _ *int, current CodecVolume[int]) (CodecVolume[int], error) {
		current.Channels = operation.OutputChannels
		return current, nil
	}); err == nil {
		t.Fatal("expected state/program mismatch refusal")
	}
}

func TestDownsampledPlanarGeometry(t *testing.T) {
	channels, height, width, err := DownsampledPlanarGeometry(16, 2048, 1024, 8)
	if err != nil {
		t.Fatal(err)
	}
	if channels != 16 || height != 256 || width != 128 {
		t.Fatalf("geometry=[%d,%d,%d]", channels, height, width)
	}
	for _, input := range [][4]int{{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 3, 4, 2}} {
		if _, _, _, err := DownsampledPlanarGeometry(input[0], input[1], input[2], input[3]); err == nil {
			t.Fatalf("accepted invalid geometry %v", input)
		}
	}
}

func TestValidatePlanarGeometry(t *testing.T) {
	if err := ValidatePlanarGeometry(4, 8, 8); err != nil {
		t.Fatal(err)
	}
	if err := ValidatePlanarGeometry(0, 8, 8); err == nil {
		t.Fatal("accepted zero channels")
	}
	if err := ValidateSpatialGeometry(0, 8); err == nil {
		t.Fatal("accepted zero height")
	}
}

func TestResidualBindings(t *testing.T) {
	base := []string{"norm", "conv"}
	if got := ResidualBindings(base, 4, 4, "projection"); len(got) != 2 {
		t.Fatalf("identity bindings=%v", got)
	}
	if got := ResidualBindings(base, 4, 8, "projection"); len(got) != 3 || got[2] != "projection" {
		t.Fatalf("projected bindings=%v", got)
	}
}

func TestCodecOperationRequiresProjection(t *testing.T) {
	identity := CodecOperation[struct{}]{Operator: CodecResidual, InputChannels: 4, OutputChannels: 4}
	projected := CodecOperation[struct{}]{Operator: CodecResidual, InputChannels: 4, OutputChannels: 8}
	if identity.RequiresProjection() || !projected.RequiresProjection() {
		t.Fatal("residual projection decision is incorrect")
	}
}

func TestConvolutionGeometryMatches(t *testing.T) {
	if !Convolution2DMatches(8, 4, 3, 3, 8, 4, 3, 3) || Convolution2DMatches(8, 4, 1, 1, 8, 4, 3, 3) {
		t.Fatal("two-dimensional convolution comparison is incorrect")
	}
	if !Convolution3DMatches(8, 4, 3, 1, 1, 8, 4, 3, 1, 1) || Convolution3DMatches(8, 4, 3, 3, 3, 8, 4, 3, 1, 1) {
		t.Fatal("three-dimensional convolution comparison is incorrect")
	}
}

func TestUpsampledCausalVolume(t *testing.T) {
	got, err := UpsampledCausalVolume(3, 4, 5, [3]int{2, 4, 4})
	if err != nil {
		t.Fatal(err)
	}
	if got != (VolumeGeometry{Frames: 5, Height: 16, Width: 20}) {
		t.Fatalf("volume = %+v", got)
	}
}

func TestVolumeGeometry(t *testing.T) {
	volume, err := DownsampledVolume(17, 64, 32, [3]int{4, 8, 8})
	if err != nil || volume != (VolumeGeometry{Frames: 5, Height: 8, Width: 4}) {
		t.Fatalf("volume=%+v err=%v", volume, err)
	}
	grid, err := VolumePatchGrid(volume, [3]int{1, 2, 2})
	if err != nil || grid != [3]int{5, 4, 2} {
		t.Fatalf("grid=%v err=%v", grid, err)
	}
	if elements, err := VolumeElements(grid); err != nil || elements != 40 {
		t.Fatalf("elements=%d err=%v", elements, err)
	}
	if _, err := DownsampledVolumeExactSpatial(5, 15, 16, [3]int{4, 8, 8}); err == nil {
		t.Fatal("accepted inexact spatial stride")
	}
}
