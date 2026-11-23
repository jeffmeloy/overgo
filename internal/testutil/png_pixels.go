package testutil

import (
	"bytes"
	"image"
	"image/draw"
	"image/png"
	"testing"

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
