package media

import (
	"errors"
	"testing"
)

// TestCodecChunkStreamOwnsCausalState holds one stream's temporal state to its
// own chunks: two streams over one program accumulate independently, a
// repeated chunk is refused, a failed operation poisons continuation until
// Reset, and Reset starts a cold stream.
func TestCodecChunkStreamOwnsCausalState(t *testing.T) {
	t.Parallel()
	op := NewCodecOperation[struct{}](CodecPointwise, "accumulate", 1, 1)
	var err error
	op.Bindings, err = BindCodecWeights(CodecPointwise, false, []struct{}{{}, {}})
	if err != nil {
		t.Fatal(err)
	}
	program := CodecProgram[struct{}]{Operations: []CodecOperation[struct{}]{op}}
	first, err := NewCodecChunkStream[struct{}, int](program)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewCodecChunkStream[struct{}, int](program)
	if err != nil {
		t.Fatal(err)
	}
	step := func(_ int, _ CodecOperation[struct{}], state *int, value CodecVolume[int]) (CodecVolume[int], error) {
		*state += value.Storage
		value.Storage = *state
		return value, nil
	}
	run := func(stream *CodecChunkStream[struct{}, int], ordinal, input, expected int) {
		t.Helper()
		value, lineage, err := ExecuteCodecChunk("state", stream, ordinal, ordinal, CodecVolume[int]{Storage: input, Channels: 1, Frames: 1, Height: 1, Width: 1}, step)
		if err != nil || value.Storage != expected || lineage != (CodecChunkLineage{Ordinal: ordinal, InputStart: ordinal, InputFrames: 1, OutputStart: ordinal, OutputFrames: 1}) {
			t.Fatalf("stream value=%+v lineage=%+v err=%v", value, lineage, err)
		}
	}
	run(first, 0, 2, 2)
	run(second, 0, 10, 10)
	run(first, 1, 3, 5)
	run(second, 1, 7, 17)
	input := CodecVolume[int]{Storage: 1, Channels: 1, Frames: 1, Height: 1, Width: 1}
	if _, _, err := ExecuteCodecChunk("duplicate", first, 1, 1, input, step); err == nil {
		t.Fatal("duplicate chunk accepted")
	}
	run(first, 2, 1, 6)
	if _, _, err := ExecuteCodecChunk("partial", first, 3, 3, input, func(_ int, _ CodecOperation[struct{}], state *int, value CodecVolume[int]) (CodecVolume[int], error) {
		*state = 999
		return value, errors.New("partial backend failure")
	}); err == nil {
		t.Fatal("partial failure disappeared")
	}
	if _, _, err := ExecuteCodecChunk("poisoned", first, 3, 3, input, step); err == nil {
		t.Fatal("failed temporal state continued")
	}
	first.Reset()
	run(first, 0, 4, 4)
	run(second, 2, 2, 19)
}
