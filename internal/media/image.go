package media

import "image"

func CloneRGBA(source *image.RGBA) *image.RGBA {
	result := image.NewRGBA(source.Bounds())
	copy(result.Pix, source.Pix)
	return result
}
