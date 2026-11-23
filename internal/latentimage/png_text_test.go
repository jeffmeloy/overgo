package latentimage

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestAPicturesTextsTravelInItsBytes holds what a picture was drawn from to its stored bytes (draw-result-carries): a
// picture saying its prompt, negative and sharpening is stored with them (PNGContent) and gives them back when its stored
// copy is read (DecodeContent), a prompt in any script among them; storing it again is the same content; and a picture
// saying nothing is stored as the bytes it was encoded as, so no other model's stored picture changes.
func TestAPicturesTextsTravelInItsBytes(t *testing.T) {
	pixels := make([]float32, 3*4*6)
	for index := range pixels {
		pixels[index] = float32(index%7) / 7
	}
	plain := testcheck.Must(EncodePlanarPNG(pixels, 4, 6))(t)
	unsaid := testcheck.Must(PNGContent(plain))(t)
	if !bytes.Equal(unsaid.Data, plain.Data) {
		t.Fatal("a picture saying nothing was stored as other bytes")
	}
	said := plain
	said.Prompt, said.NegativePrompt, said.Sharpened = `{"summary":{"caption":"蓝色方块"}} a fox`, "grey sky", true
	stored := testcheck.Must(PNGContent(said))(t)
	var read EncodedImage
	testcheck.NoError(t, read.DecodeContent(stored))
	if read.Prompt != said.Prompt || read.NegativePrompt != said.NegativePrompt || !read.Sharpened || read.Width != 6 || read.Height != 4 {
		t.Errorf("the stored picture gave back %q, %q, sharpened %t, %dx%d", read.Prompt, read.NegativePrompt, read.Sharpened, read.Width, read.Height)
	}
	if again := testcheck.Must(PNGContent(read))(t); again.Descriptor.ID != stored.Descriptor.ID {
		t.Errorf("the picture stored again is %s; want %s", again.Descriptor.ID, stored.Descriptor.ID)
	}
	read.Prompt, read.NegativePrompt, read.Sharpened = "", "", false
	if cleared := testcheck.Must(PNGContent(read))(t); !bytes.Equal(cleared.Data, plain.Data) {
		t.Error("a picture whose texts were taken out is not the picture it was encoded as")
	}
	id := testcheck.Must(artifact.IdentifyBytes(artifact.KindOutput, stored.Data))(t)
	if id != stored.Descriptor.ID {
		t.Errorf("the stored picture is named %s by its bytes; want %s", id, stored.Descriptor.ID)
	}
}
