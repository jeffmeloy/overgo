//go:build windows

package mediacapability

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// interruptedPair is a store whose first commit of a sharpened picture's pair (keptDrawn's "picture-drawn/" batch)
// fails, as a process stopped after its run kept the sharpened picture and before the pair was kept.
type interruptedPair struct {
	countedStore
	interrupted bool
}

// Commit fails the first pair's batch and commits the rest.
func (s *interruptedPair) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	if !s.interrupted && strings.HasPrefix(batch.Key, "picture-drawn/") {
		s.interrupted = true
		return artifact.CommitID{}, errors.New("the process stopped before the pair was kept")
	}
	return s.countedStore.Commit(ctx, batch)
}

// TestAStoredSharpenedPictureWithoutItsDrawnOneIsDrawnAgain holds served-answers/pair (SQA-466): a sharpened picture
// whose run was kept but whose pair was not, the process stopped between the two, is drawn again when the request is
// served again, once, and comes with the picture it drew; the pair now kept, the request is answered by a stored run
// with both pictures and no draw. keptDrawn had given the stored sharpened picture alone, as a success, at every repeat.
func TestAStoredSharpenedPictureWithoutItsDrawnOneIsDrawnAgain(t *testing.T) {
	pictures := newStandInPictures(t, enhancerOpener)
	const sharpened = `{"prompt":"a lighthouse","seed":5,"width":256,"height":144,"sharpen":{"radius":1,"sigma":0.6,"alpha":0.7}}`
	stopping := &interruptedPair{countedStore: pictures.store}
	if _, err := pictures.serve(t.Context(), stopping, pictures.root, pictures.execution, sharpened); err == nil || !stopping.interrupted {
		t.Fatalf("the interrupted serving gave %v; want its pair's keeping to stop it", err)
	}
	draws := pictures.work.draws
	measured, picture := pictures.served(t.Context(), sharpened)
	if pictures.work.draws != draws+1 || len(picture.Drawn) == 0 || bytes.Equal(picture.Drawn, picture.Data) {
		t.Fatalf("the repeat drew %d times and came with %d bytes as the picture drawn; want one draw and the picture before its sharpening",
			pictures.work.draws-draws, len(picture.Drawn))
	}
	// Either stored run may answer the request: the picture drawn again is the interrupted run's bytes, so its pair is both's.
	again, repeat := pictures.served(t.Context(), sharpened)
	if pictures.work.draws != draws+1 || !again.Run.Valid() || !bytes.Equal(repeat.Drawn, picture.Drawn) || !bytes.Equal(repeat.Data, picture.Data) {
		t.Errorf("the pair kept, the request drew %d times more and came with %d bytes as the picture drawn; want a stored run's pair and no draw",
			pictures.work.draws-draws-1, len(repeat.Drawn))
	}
	if measured.Run == again.Run && measured.Load != again.Load {
		t.Errorf("one run measured two loads, %v and %v", measured.Load, again.Load)
	}
}
