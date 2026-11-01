//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/graphruntime"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func verifyMOSSDecoder(ctx context.Context, runtime *graphruntime.ResidentSession, modelRoot, referenceRoot, name string, identity artifact.ID, source *safetensors.Source, tolerances map[string]mossStageTolerance, output io.Writer) error {
	d, err := loadSpatialVAEDecoder(modelRoot)
	if err != nil {
		return err
	}
	schedule, ok := source.Tensors["schedule.timesteps"]
	if !ok || schedule.Elements() == 0 {
		return fmt.Errorf("MOSS decoder reference has no schedule")
	}
	latent, ok := source.Tensors[fmt.Sprintf("latents.%d", schedule.Elements()-1)]
	if !ok || len(latent.Shape) != rotaryCoordinateAxes || latent.Shape[0] != tensor.SingletonExtent {
		return fmt.Errorf("MOSS decoder final latent shape differs")
	}
	p, err := CompileVAEProgram(d, int(latent.Shape[tensor.PairedExtent]), int(latent.Shape[tensor.TripleExtent]), dtype.F32)
	if err != nil {
		return err
	}
	values, err := safetensors.ReadF32(latent)
	if err != nil {
		return err
	}
	inputElements, err := p.Latent.Shape.Elements()
	if err != nil || uint64(len(values)) != latent.Elements() || latent.Elements() != inputElements {
		return fmt.Errorf("MOSS decoder latent payload differs")
	}
	named := map[string]*tensor.Tensor{"vae.unpacked": p.boundaries["unpacked"], "vae.decoded": p.decoded, "pixels": p.pixels}
	expected := map[string]safetensors.Tensor{}
	references := map[string]artifact.ID{}
	for key := range named {
		v, ok := source.Tensors[key]
		if !ok {
			return fmt.Errorf("MOSS decoder reference lacks %s", key)
		}
		expected[key] = v
		references[key] = identity
	}
	if name == "full-english" {
		path := filepath.Join(referenceRoot, "decoder-english.safetensors")
		id, e := artifact.IdentifyFile(artifact.KindFile, path)
		if e != nil {
			return e
		}
		if id.DigestHex() != "ca8c61f787085e268202e098970cb1a58f2df6029e856d6ccd8786372e140314" {
			return fmt.Errorf("MOSS native decoder boundary identity differs")
		}
		authority, e := safetensors.OpenFile(path)
		if e != nil {
			return e
		}
		defer authority.Close()
		for key, v := range authority.Tensors {
			if key == "input" || key == "decoded" {
				continue
			}
			node := p.boundaries[key]
			if key == "decoder.conv_norm_out" {
				node = p.boundaries["decoder.conv_out.norm"]
			}
			if key == "decoder.mid_block" || strings.HasPrefix(key, "decoder.up_blocks.") {
				for _, op := range d.Operations {
					if strings.HasPrefix(op.Name, key+".") {
						node = p.boundaries[op.Name]
					}
				}
			}
			if node == nil {
				return fmt.Errorf("MOSS decoder has no boundary %s", key)
			}
			named[key] = node
			expected[key] = v
			references[key] = id
		}
	}
	keys := make([]string, 0, len(named))
	for key := range named {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	var outputs []*tensor.Tensor
	seen := map[*tensor.Tensor]bool{}
	for _, key := range keys {
		node := named[key]
		if !seen[node] {
			outputs = append(outputs, node)
			seen[node] = true
		}
	}
	static := map[*tensor.Tensor][]float32{}
	for _, feed := range p.feeds {
		static[feed.node] = feed.data
	}
	program, err := compileResidentStatic(ctx, runtime, "MOSS decoder "+modelRoot, static, nil, outputs...)
	if err != nil {
		return err
	}
	defer runtime.Release(context.WithoutCancel(ctx), program)
	feeds := map[*tensor.Tensor]reference.Value{p.Latent: {Shape: p.Latent.Shape, Data: values}}
	hashes := map[string][sha256.Size]byte{}
	failures := 0
	for pass := range tensor.PairedExtent {
		actual, e := runtime.Execute(ctx, program, feeds)
		if e != nil {
			return e
		}
		for _, key := range keys {
			node := named[key]
			hwc := actual[node].Data
			c := int(node.Shape.Dims[0])
			plane := len(hwc) / c
			planar := make([]float32, len(hwc))
			for pos := range plane {
				for ch := range c {
					planar[ch*plane+pos] = hwc[pos*c+ch]
				}
			}
			want, e := safetensors.ReadF32(expected[key])
			if e != nil {
				return e
			}
			tolerance, ok := tolerances[expected[key].DType]
			if !ok {
				return fmt.Errorf("MOSS decoder reference dtype has no tolerance")
			}
			failed, unequal, worst := compareMOSSValues(planar, want, tolerance)
			hash := sha256.Sum256(driver.Bytes(planar))
			repeatEqual := true
			if pass == 0 {
				hashes[key] = hash
			} else {
				repeatEqual = hashes[key] == hash
			}
			var encoded *artifact.Descriptor
			if key == "pixels" {
				image, e := encodeUnitPNG(hwc, p.OutH, p.OutW)
				if e != nil {
					return e
				}
				content, e := PNGContent(image)
				if e != nil {
					return e
				}
				encoded = &content.Descriptor
				imageHash := sha256.Sum256(content.Data)
				if pass == 0 {
					hashes["png"] = imageHash
				} else {
					repeatEqual = repeatEqual && hashes["png"] == imageHash
				}
			}
			if e := json.NewEncoder(output).Encode(map[string]any{"stage": "decoder", "case": name, "boundary": key, "reference": references[key], "pass": pass, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeatEqual, "resident_weight_and_static_bytes": runtime.StaticBytes(), "encoded_image": encoded}); e != nil {
				return e
			}
			failures += failed
			if !repeatEqual {
				failures++
			}
		}
	}
	if failures != 0 {
		return fmt.Errorf("MOSS decoder violates frozen comparison in %s (%d failures)", name, failures)
	}
	return nil
}
