package oscillatorimage

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
)

type VideoRequest struct {
	Class  int   `json:"class"`
	Seed   int64 `json:"seed"`
	Frames int   `json:"frames"`
	Scale  int   `json:"scale"`
}

type EncodedVideo struct {
	Data          []byte `json:"data"`
	MediaType     string `json:"media_type"`
	Frames        int    `json:"frames"`
	Channels      int    `json:"channels"`
	Height        int    `json:"height"`
	Width         int    `json:"width"`
	ChangedPixels int    `json:"changed_pixels"`
}

type videoPlan struct{ request VideoRequest }
type videoFeatures struct {
	frames [][]float32
	scale  int
}

func ValidateVideoRequest(request VideoRequest) error {
	if request.Class < 0 || request.Frames <= 0 || request.Scale <= 0 {
		return errors.New("oscillatorimage: video requires non-negative class and positive frame count and scale")
	}
	return nil
}

func (m *Model) prepareVideo(request VideoRequest) (videoPlan, error) {
	if err := ValidateVideoRequest(request); err != nil {
		return videoPlan{}, err
	}
	if m == nil || request.Class >= m.Cfg.NClasses {
		return videoPlan{}, errors.New("oscillatorimage: video class outside artifact")
	}
	return videoPlan{request: request}, nil
}

func (m *Model) integrateVideo(plan videoPlan) (videoFeatures, error) {
	features := videoFeatures{frames: make([][]float32, plan.request.Frames), scale: plan.request.Scale}
	for frame := range features.frames {
		phase, err := m.prepare(Request{Class: plan.request.Class, Seed: plan.request.Seed + int64(frame)})
		if err != nil {
			return videoFeatures{}, err
		}
		features.frames[frame], err = m.integrate(phase)
		if err != nil {
			return videoFeatures{}, err
		}
	}
	return features, nil
}

func (m *Model) decodeVideo(features videoFeatures) (EncodedVideo, error) {
	if len(features.frames) == 0 {
		return EncodedVideo{}, errors.New("oscillatorimage: video features are empty")
	}
	animation := &gif.GIF{}
	var previous []uint8
	changed := 0
	for _, feature := range features.frames {
		frame, err := m.decodePlanar(feature)
		if err != nil {
			return EncodedVideo{}, err
		}
		rgba, raw, err := planarRGBA(frame, features.scale)
		if err != nil {
			return EncodedVideo{}, err
		}
		if previous != nil {
			for index := 0; index < len(raw); index += 3 {
				if raw[index] != previous[index] || raw[index+1] != previous[index+1] || raw[index+2] != previous[index+2] {
					changed++
				}
			}
		}
		previous = raw
		paletted := image.NewPaletted(rgba.Bounds(), palette.Plan9)
		draw.FloydSteinberg.Draw(paletted, rgba.Bounds(), rgba, image.Point{})
		animation.Image = append(animation.Image, paletted)
		animation.Delay = append(animation.Delay, 12)
	}
	var encoded bytes.Buffer
	if err := gif.EncodeAll(&encoded, animation); err != nil {
		return EncodedVideo{}, err
	}
	return EncodedVideo{
		Data: encoded.Bytes(), MediaType: "image/gif", Frames: len(features.frames), Channels: m.Cfg.OutChannels,
		Height: m.Cfg.OutH() * features.scale, Width: m.Cfg.OutW() * features.scale, ChangedPixels: changed,
	}, nil
}

func planarRGBA(frame planarImage, scale int) (*image.RGBA, []uint8, error) {
	if frame.Channels != 3 || len(frame.Pixels) != frame.Channels*frame.Height*frame.Width {
		return nil, nil, errors.New("oscillatorimage: video frame is not planar RGB")
	}
	if scale <= 0 {
		return nil, nil, errors.New("oscillatorimage: video scale is invalid")
	}
	width, height := frame.Width*scale, frame.Height*scale
	output := image.NewRGBA(image.Rect(0, 0, width, height))
	raw := make([]uint8, frame.Width*frame.Height*3)
	plane := frame.Width * frame.Height
	pixel := func(value float32) uint8 {
		return uint8(min(max((float64(value)+1)*127.5, 0), 255))
	}
	for y := range frame.Height {
		for x := range frame.Width {
			source := y*frame.Width + x
			r, g, b := pixel(frame.Pixels[source]), pixel(frame.Pixels[plane+source]), pixel(frame.Pixels[2*plane+source])
			raw[source*3], raw[source*3+1], raw[source*3+2] = r, g, b
			for yy := range scale {
				for xx := range scale {
					output.SetRGBA(x*scale+xx, y*scale+yy, color.RGBA{R: r, G: g, B: b, A: 255})
				}
			}
		}
	}
	return output, raw, nil
}
