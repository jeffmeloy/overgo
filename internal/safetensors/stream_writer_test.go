package safetensors

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/binaryschema"
	"overgo/internal/testcheck"
)

func TestStreamingWriterDeterministicShards(t *testing.T) {
	plan := streamPlan{Shards: []streamShardSpec{
		{Name: "model-b.safetensors", Tensors: []streamTensorSpec{{Name: "b", DType: "U8", Shape: []uint64{3}}}},
		{Name: "model-a.safetensors", Tensors: []streamTensorSpec{{Name: "a", DType: "U8", Shape: []uint64{2}}}},
	}, Metadata: map[string]string{"format": "pt"}}
	first := filepath.Join(t.TempDir(), "first")
	writer := testcheck.Must(newStreamWriter(first, plan))(t)
	testcheck.NoError(t, writer.writeTensor("b", 3, bytes.NewReader([]byte{3, 4, 5})))
	testcheck.NoError(t, writer.writeTensor("a", 2, bytes.NewReader([]byte{1, 2})))
	testcheck.NoError(t, writer.finalize())

	second := filepath.Join(t.TempDir(), "second")
	writer = testcheck.Must(newStreamWriter(second, streamPlan{Shards: []streamShardSpec{plan.Shards[1], plan.Shards[0]}, Metadata: plan.Metadata}))(t)
	testcheck.NoError(t, writer.writeTensor("a", 2, bytes.NewReader([]byte{1, 2})))
	testcheck.NoError(t, writer.writeTensor("b", 3, bytes.NewReader([]byte{3, 4, 5})))
	testcheck.NoError(t, writer.finalize())
	for _, name := range []string{"model-a.safetensors", "model-b.safetensors", streamIndexName} {
		left := testcheck.Must(os.ReadFile(filepath.Join(first, name)))(t)
		right := testcheck.Must(os.ReadFile(filepath.Join(second, name)))(t)
		if !bytes.Equal(left, right) {
			t.Fatalf("streaming artifact %s differs across plan and write order", name)
		}
	}
}

func TestStreamingWriterAtomicFinalize(t *testing.T) {
	destination := filepath.Join(t.TempDir(), "model")
	plan := streamPlan{Shards: []streamShardSpec{{
		Name: "model.safetensors", Tensors: []streamTensorSpec{{Name: "weight", DType: "U8", Shape: []uint64{2}}},
	}}}
	writer := testcheck.Must(newStreamWriter(destination, plan))(t)
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("destination visible before finalization: %v", err)
	}
	if err := writer.writeTensor("weight", 2, bytes.NewReader([]byte{1})); err == nil || len(writer.completed) != 0 {
		t.Fatalf("short payload accepted or sealed: %v", err)
	}
	if err := writer.finalize(); err == nil {
		t.Fatal("finalized an incomplete streaming artifact")
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("incomplete destination became visible: %v", err)
	}
	testcheck.NoError(t, writer.writeTensor("weight", 2, bytes.NewReader([]byte{1, 2})))
	testcheck.NoError(t, writer.finalize())
	if _, err := os.Stat(filepath.Join(destination, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(destination + streamStagingSuffix); !os.IsNotExist(err) {
		t.Fatalf("staging directory remains after finalization: %v", err)
	}
}

func TestStreamingWriterResume(t *testing.T) {
	destination := filepath.Join(t.TempDir(), "model")
	plan := streamPlan{Shards: []streamShardSpec{{
		Name: "model.safetensors", Tensors: []streamTensorSpec{
			{Name: "first", DType: "U8", Shape: []uint64{2}},
			{Name: "second", DType: "U8", Shape: []uint64{3}},
		},
	}}}
	writer := testcheck.Must(newStreamWriter(destination, plan))(t)
	testcheck.NoError(t, writer.writeTensor("first", 2, bytes.NewReader([]byte{1, 2})))
	resumed := testcheck.Must(newStreamWriter(destination, plan))(t)
	if len(resumed.completed) != 1 {
		t.Fatalf("resumed completion count = %d", len(resumed.completed))
	}
	if err := resumed.writeTensor("first", 2, bytes.NewReader([]byte{1, 2})); err == nil {
		t.Fatal("rewrote a completed tensor after resume")
	}
	testcheck.NoError(t, resumed.writeTensor("second", 3, bytes.NewReader([]byte{3, 4, 5})))
	testcheck.NoError(t, resumed.finalize())
	source := testcheck.Must(OpenSource(destination))(t)
	defer source.Close()
	for name, want := range map[string][]byte{"first": {1, 2}, "second": {3, 4, 5}} {
		got := testcheck.Must(io.ReadAll(source.Tensors[name].Reader()))(t)
		if !bytes.Equal(got, want) {
			t.Fatalf("tensor %s = %v, want %v", name, got, want)
		}
	}
}

func TestStreamingWriterDerivedBounds(t *testing.T) {
	destination := filepath.Join(t.TempDir(), "model")
	plan := streamPlan{Shards: []streamShardSpec{{
		Name: "model.safetensors", Tensors: []streamTensorSpec{{Name: "matrix", DType: "F32", Shape: []uint64{2, 2}}, {Name: "empty", DType: "U8", Shape: []uint64{0}}},
	}}}
	writer := testcheck.Must(newStreamWriter(destination, plan))(t)
	testcheck.NoError(t, writer.writeTensor("empty", 0, bytes.NewReader(nil)))
	payload := floatPayload(1, 2, 3, 4)
	if err := writer.writeTensor("matrix", uint64(len(payload)-1), bytes.NewReader(payload)); err == nil {
		t.Fatal("accepted caller extent that differs from dtype and shape")
	}
	testcheck.NoError(t, writer.writeTensor("matrix", uint64(len(payload)), bytes.NewReader(payload)))
	testcheck.NoError(t, writer.finalize())
	source := testcheck.Must(OpenSource(destination))(t)
	defer source.Close()
	tensor := source.Tensors["matrix"]
	if tensor.Size() != int64(len(payload)) || tensor.Shape[0] != 2 || tensor.Shape[1] != 2 {
		t.Fatalf("derived tensor bounds = size %d shape %v", tensor.Size(), tensor.Shape)
	}
}

func floatPayload(values ...float32) []byte {
	return binaryschema.LittleEndian.Float32s(values)
}
