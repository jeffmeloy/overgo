package projector

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
)

func openImageProjectorAs[T Projector](path string, options OpenOptions) (T, error) {
	return OpenAs[T](context.Background(), path, options)
}

// testSession compiles the prompt session for a hermetic runner fixture.
func testSession(t *testing.T, source Projector) Session {
	t.Helper()
	session, err := NewSession(source)
	if err != nil {
		t.Fatal(err)
	}
	return session
}

func TestOpenProjectorResourceClosesFileOnBuildFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "projector.gguf")
	output, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(output, nil, []gguf.TensorData{
		f32Tensor("weight", []uint64{1}, []float32{1}),
	}, gguf.WriteOptions{}); err != nil {
		_ = output.Close()
		t.Fatal(err)
	}
	if err := output.Close(); err != nil {
		t.Fatal(err)
	}
	want := errors.New("build failed")
	var captured *gguf.File
	result, err := openProjectorResource(t.Context(), path, func(file *gguf.File) (*gguf.File, error) {
		captured = file
		return nil, want
	})
	if result != nil || !errors.Is(err, want) {
		t.Fatalf("result/error = %v/%v", result, err)
	}
	info, ok := captured.Tensor("weight")
	if !ok {
		t.Fatal("captured file lost tensor catalog")
	}
	if err := captured.ReadTensorData(info, make([]byte, info.Size)); err == nil {
		t.Fatal("failed build left GGUF handle open")
	}
}

func TestOpenProjectorResourceRejectsCanceledContextBeforeOpen(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	called := false
	_, err := openProjectorResource(ctx, "missing.gguf", func(*gguf.File) (struct{}, error) {
		called = true
		return struct{}{}, nil
	})
	if !errors.Is(err, context.Canceled) || called {
		t.Fatalf("error/build = %v/%t", err, called)
	}
}
