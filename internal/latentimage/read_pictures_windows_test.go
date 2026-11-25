//go:build windows

package latentimage

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/png"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestAStoredOutputPictureIsReadAsAPicture holds moss-pictures/output-input: a request's picture named by a stored
// output's identity, as a page's use as input names a MOSS run's picture, is read into the bytes its session takes, as
// the same PNG stored as a file is, for its picture to sample from and its depth control's picture alike; an output
// that is no picture, and an output not stored, are refused. ReadPictures had refused every output as not a stored file
// before reading it, so a MOSS picture used as input was refused where the same bytes uploaded passed (SQA-492).
func TestAStoredOutputPictureIsReadAsAPicture(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, image.NewNRGBA(image.Rect(0, 0, 8, 6))))
	picture := encoded.Bytes()
	stored := func(kind artifact.Kind, data []byte, mediaType string) artifact.ID {
		t.Helper()
		content := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(kind, data))(t), Size: uint64(len(data)), MediaType: mediaType}, Data: data}
		testcheck.Must(artifact.Publish(t.Context(), store, artifact.Batch{Key: "fixture/" + content.Descriptor.ID.String(), Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content}}))(t)
		return content.Descriptor.ID
	}
	want := base64.StdEncoding.EncodeToString(picture)
	for _, id := range []artifact.ID{stored(artifact.KindFile, picture, "image/png"), stored(artifact.KindOutput, picture, "image/png")} {
		request := Request{Image: id.String(), Control: &DepthControl{Picture: id.String()}}
		read, err := request.ReadPictures(t.Context(), store)
		if err != nil || read.Image != want || read.Control.Picture != want {
			t.Errorf("a request naming the %s %s read as %.20q and %.20q (%v); want the stored PNG's bytes for both", id.Kind(), id, read.Image, read.Control.Picture, err)
		}
	}
	document := stored(artifact.KindOutput, []byte(`{"text":"no picture"}`), "application/json")
	if _, err := (Request{Image: document.String()}).ReadPictures(t.Context(), store); err == nil || !strings.Contains(err.Error(), "not a picture") {
		t.Errorf("a request naming an output that is no picture read as %v; want it refused", err)
	}
	absent := testcheck.Must(artifact.IdentifyBytes(artifact.KindOutput, []byte("never stored")))(t)
	if _, err := (Request{Image: absent.String()}).ReadPictures(t.Context(), store); err == nil || !strings.Contains(err.Error(), "is not stored") {
		t.Errorf("a request naming an output never stored read as %v; want it refused", err)
	}
}
