package projector

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"image"
	"sync"
)

// rememberedImages is how many encoded pictures a projector session keeps: a chat sends its pictures again with every
// turn, and a picture sent again is encoded once.
const rememberedImages = 4

// rememberedImage is one encoded picture, found by the digest of its pixels.
type rememberedImage struct {
	pixels [sha256.Size]byte
	item   imagePromptItem
}

// rememberImages keeps what encode made of the last pictures it was given, found by their pixels, so a picture sent
// again is not read, resized or encoded again. An item is shared, which its readers copy and never change.
func rememberImages(encode imagePromptEncoder) imagePromptEncoder {
	if encode == nil {
		return nil
	}
	var mu sync.Mutex
	var kept []rememberedImage
	return func(ctx context.Context, source image.Image) (imagePromptItem, error) {
		pixels := pixelDigest(source)
		mu.Lock()
		for index, entry := range kept {
			if entry.pixels == pixels {
				// The picture asked for last is the last to go.
				kept = append(append(kept[:index:index], kept[index+1:]...), entry)
				mu.Unlock()
				return entry.item, nil
			}
		}
		mu.Unlock()
		item, err := encode(ctx, source)
		if err != nil {
			return item, err
		}
		mu.Lock()
		defer mu.Unlock()
		kept = append(kept, rememberedImage{pixels: pixels, item: item})
		if len(kept) > rememberedImages {
			kept = kept[1:]
		}
		return item, nil
	}
}

// pixelDigest names a picture by its bounds and pixels: an RGBA or NRGBA picture's own bytes, a YCbCr picture's
// planes, and any other picture's colors read through At.
func pixelDigest(source image.Image) [sha256.Size]byte {
	digest := sha256.New()
	bounds := source.Bounds()
	fmt.Fprintf(digest, "%T %v\n", source, bounds)
	switch picture := source.(type) {
	case *image.RGBA:
		digest.Write(picture.Pix)
	case *image.NRGBA:
		digest.Write(picture.Pix)
	case *image.YCbCr:
		fmt.Fprintln(digest, picture.SubsampleRatio)
		digest.Write(picture.Y)
		digest.Write(picture.Cb)
		digest.Write(picture.Cr)
	default:
		var row []byte
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			row = row[:0]
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				r, g, b, a := source.At(x, y).RGBA()
				for _, channel := range [...]uint32{r, g, b, a} {
					row = binary.LittleEndian.AppendUint16(row, uint16(channel))
				}
			}
			digest.Write(row)
		}
	}
	var sum [sha256.Size]byte
	copy(sum[:], digest.Sum(nil))
	return sum
}
