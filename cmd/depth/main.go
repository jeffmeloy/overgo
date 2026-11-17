// Command depth writes the relative depth map of a picture, white for near, as the MOSS Ultra workflow makes it:
// Depth Anything V2 Small run as comfyui_controlnet_aux's DepthAnythingV2Preprocessor runs it at a detect resolution
// (vitencoder.DepthMap), then scaled back to the picture's size bilinearly as ComfyUI's ImageScale scales it, and
// saved as ComfyUI saves a picture, 255 x truncated to 8 bits.
//
//	depth -in picture.png -out depth.png [-resolution 512] [-model depth-anything-v2-small/depth_anything_v2_vits.pth]
package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/media"
	"overgo/internal/vitencoder"
)

func main() {
	clioptions.Main(run)
}

func run() error {
	in := flag.String("in", "", "the picture, a PNG")
	out := flag.String("out", "", "where the depth map is written, a PNG")
	resolution := flag.Int("resolution", 512, "the detect resolution: the shorter side the preprocessor sizes its map to, as the workflow's node takes it")
	model := flag.String("model", "depth-anything-v2-small/depth_anything_v2_vits.pth", "the Depth Anything V2 checkpoint, a path or a model reference under the data root")
	flag.Parse()
	if *in == "" || *out == "" {
		return errors.New("depth: -in and -out name the picture and the map")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	data, err := os.ReadFile(*in)
	if err != nil {
		return err
	}
	picture, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("depth: %s is not a PNG: %w", *in, err)
	}
	bounds := picture.Bounds()
	height, width := bounds.Dy(), bounds.Dx()
	// The picture as ComfyUI's LoadImage holds it: 8-bit RGB over 255.
	unit := make([]float32, 0, height*width*media.RGBChannels)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixel := color.NRGBAModel.Convert(picture.At(x, y)).(color.NRGBA)
			unit = append(unit, float32(pixel.R)/media.UnitLevels, float32(pixel.G)/media.UnitLevels, float32(pixel.B)/media.UnitLevels)
		}
	}
	depth, err := vitencoder.LoadDepthAnything(roots.ResolveModelPath(*model))
	if err != nil {
		return err
	}
	levels, mapHeight, mapWidth, err := depth.DepthMap(unit, height, width, *resolution)
	if err != nil {
		return err
	}
	shown := make([]float32, len(levels))
	for index, level := range levels {
		shown[index] = float32(level) / media.UnitLevels
	}
	scaled, err := media.BilinearLikeComfyUI(shown, mapHeight, mapWidth, 1, height, width)
	if err != nil {
		return err
	}
	written := image.NewGray(image.Rect(0, 0, width, height))
	for index, value := range scaled {
		written.Pix[index] = uint8(min(max(value*media.UnitLevels, 0), media.UnitLevels))
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, written); err != nil {
		return err
	}
	if err := os.WriteFile(*out, encoded.Bytes(), 0o644); err != nil {
		return err
	}
	_, err = fmt.Printf("depth: %s, %d x %d, its map %d x %d at resolution %d, written to %s\n", *in, width, height, mapWidth, mapHeight, *resolution, *out)
	return err
}
