package dataset

import (
	"context"
	"errors"
	"os"
	"sync"
	"testing"

	"overgo/internal/artifact"
)

func TestAudioPayloadReaderCoordinatesAndIdentity(t *testing.T) {
	path, budget := nullableParquetFixture(t)
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	container, _, err := artifact.Identify(artifact.KindDatasetShard, file)
	_ = file.Close()
	if err != nil {
		t.Fatal(err)
	}
	source, err := NewAudioPayloadReader(budget)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	audio, _ := artifact.IdentifyBytes(artifact.KindFile, []byte("e"))
	reference := AudioPayloadReference{Path: path, Audio: audio, Origin: AudioPayloadOrigin{Container: container, Column: "audio.bytes", Row: new(uint64(4))}}
	if data, err := source.Read(t.Context(), reference, 1); err != nil || string(data) != "e" {
		t.Fatalf("physical: %q %v", data, err)
	}
	var readers sync.WaitGroup
	for index, want := range []string{"e", "f"} {
		readers.Go(func() {
			id, _ := artifact.IdentifyBytes(artifact.KindFile, []byte(want))
			ref := AudioPayloadReference{Path: path, Audio: id, Origin: AudioPayloadOrigin{Container: container, Column: "audio.bytes", Row: new(uint64(index + 4))}}
			data, err := source.Read(t.Context(), ref, 1)
			if err != nil || string(data) != want {
				t.Errorf("concurrent read: %q %v", data, err)
			}
		})
	}
	readers.Wait()
	reference.Origin.Row = nil
	reference.Origin.ValueIndex = 2 // non-null empty value at ordinal 1 still counts
	if data, err := source.Read(t.Context(), reference, 1); err != nil || string(data) != "e" {
		t.Fatalf("legacy: %q %v", data, err)
	}
	reference.Origin.Row = new(uint64(4))
	if _, err := source.Read(t.Context(), reference, 1); err == nil {
		t.Fatal("ambiguous row and value index admitted")
	}
	reference.Origin.ValueIndex = 0
	*reference.Origin.Row = 1
	if _, err := source.Read(t.Context(), reference, 1); err == nil {
		t.Fatal("null audio admitted")
	}
	*reference.Origin.Row = 0
	if _, err := source.Read(t.Context(), reference, 1); err == nil {
		t.Fatal("wrong payload digest admitted")
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := source.Read(ctx, reference, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
	reference.Origin.Container = audio
	if _, err := source.Read(t.Context(), reference, 1); err == nil {
		t.Fatal("wrong container digest admitted")
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := source.Read(t.Context(), reference, 1); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("closed: %v", err)
	}
}
