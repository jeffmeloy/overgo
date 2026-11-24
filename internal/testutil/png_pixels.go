package testutil

import (
	"bytes"
	"image"
	"image/draw"
	"image/png"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// CanonicalPNG is the PNG image/png writes at its default level for the pixels encoded decodes to, as opaque RGBA: one
// set of pixels has one canonical PNG, whatever level and filters encoded was written at. A check pinning a picture's
// pixels by a PNG's digest pins this one (png-level wrote MOSS's pictures at deflate level 4, their pixels unchanged).
func CanonicalPNG(t testing.TB, encoded []byte) []byte {
	t.Helper()
	decoded := testcheck.Must(png.Decode(bytes.NewReader(encoded)))(t)
	canvas := image.NewRGBA(decoded.Bounds())
	draw.Draw(canvas, canvas.Bounds(), decoded, decoded.Bounds().Min, draw.Src)
	var written bytes.Buffer
	testcheck.NoError(t, png.Encode(&written, canvas))
	return written.Bytes()
}

// CanonicalDigest is the hex digest an output artifact of encoded's canonical PNG (CanonicalPNG) is identified by: a
// frozen picture's output identity, recorded when its PNG was the canonical one, held by its pixels whatever level it is
// now written at (one-png-writer).
func CanonicalDigest(t testing.TB, encoded []byte) string {
	t.Helper()
	return testcheck.Must(artifact.IdentifyBytes(artifact.KindOutput, CanonicalPNG(t, encoded)))(t).DigestHex()
}
