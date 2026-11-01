//go:build windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

func verifyMOSSGeneration(ctx context.Context, root, name string, id artifact.ID, source *safetensors.Source, tolerances map[string]mossStageTolerance, output io.Writer) error {
	metadata := map[string]string{}
	for _, m := range source.Metadata {
		metadata = m
	}
	request := Request{Prompt: metadata["prompt"], NegativePrompt: metadata["negative_prompt"]}
	var err error
	for key, target := range map[string]*int{"width": &request.Width, "height": &request.Height, "steps": &request.Steps} {
		*target, err = strconv.Atoi(metadata[key])
		if err != nil {
			return err
		}
	}
	request.Seed, err = strconv.ParseInt(metadata["seed"], mossMetadataRadix, mossSeedBits)
	if err != nil {
		return err
	}
	return compareGeneration(ctx, root, name, "", request, id, source, tolerances, output)
}

func compareGeneration(ctx context.Context, root, name, prefix string, request Request, id artifact.ID, source *safetensors.Source, tolerances map[string]mossStageTolerance, output io.Writer) error {
	profile, err := ResolveProfile(root)
	if err != nil {
		return err
	}
	generator, err := LoadGenerator(ctx, root, profile, request)
	if err != nil {
		return err
	}
	defer generator.Close(context.WithoutCancel(ctx))
	hashes := map[string][sha256.Size]byte{}
	var imageHash [sha256.Size]byte
	for pass := range tensor.PairedExtent {
		if pass != 0 {
			if err = generator.Reset(ctx, request); err != nil {
				return err
			}
		}
		generator.observe = func(boundary string, actual []float32) error {
			item, ok := source.Tensors[prefix+boundary]
			if !ok {
				return fmt.Errorf("generation reference lacks %s", boundary)
			}
			var want []float32
			tolerance, ok := tolerances[item.DType]
			if item.DType == "I64" {
				raw, e := io.ReadAll(item.Reader())
				if e != nil {
					return e
				}
				size := binary.Size(int64(0))
				if len(raw) != len(actual)*size {
					return fmt.Errorf("generation token extent differs")
				}
				want = make([]float32, len(actual))
				for i := range want {
					want[i] = float32(int64(binary.LittleEndian.Uint64(raw[i*size:])))
				}
				tolerance, ok = mossStageTolerance{}, true
			} else {
				var e error
				want, e = safetensors.ReadF32(item)
				if e != nil {
					return e
				}
			}
			if !ok {
				return fmt.Errorf("generation tolerance missing for %s", item.DType)
			}
			if boundary == "initial_latents" || strings.HasPrefix(boundary, "schedule.") || strings.HasPrefix(boundary, "conditioning.") {
				tolerance = mossStageTolerance{}
			}
			failed, unequal, worst := compareMOSSValues(actual, want, tolerance)
			hash := sha256.Sum256(driver.Bytes(actual))
			previous, seen := hashes[boundary]
			repeat := pass == 0 || seen && previous == hash
			hashes[boundary] = hash
			if err := json.NewEncoder(output).Encode(map[string]any{"stage": "generation", "case": name, "boundary": boundary, "reference": id, "pass": pass, "values": len(want), "failed": failed, "unequal": unequal, "max_abs": worst, "repeat_checked": pass != 0, "repeat_equal": repeat}); err != nil {
				return err
			}
			if failed != 0 || !repeat {
				return fmt.Errorf("MOSS generation violates frozen comparison in %s/%s", name, boundary)
			}
			return nil
		}
		session, e := generator.prepare(ctx, request)
		if e != nil {
			return e
		}
		session, e = generator.integrate(ctx, session)
		if e != nil {
			return e
		}
		image, e := generator.decode(ctx, session)
		if e != nil {
			return e
		}
		content, e := PNGContent(image)
		if e != nil {
			return e
		}
		hash := sha256.Sum256(content.Data)
		if pass != 0 && imageHash != hash {
			return fmt.Errorf("MOSS generation PNG repeat differs")
		}
		imageHash = hash
		if err = json.NewEncoder(output).Encode(map[string]any{"stage": "generation-image", "case": name, "pass": pass, "image": content.Descriptor.ID, "repeat_equal": true, "failed": 0}); err != nil {
			return err
		}
	}
	for key := range source.Tensors {
		boundary, ok := strings.CutPrefix(key, prefix)
		if !ok {
			continue
		}
		if strings.HasPrefix(boundary, "prediction.") || strings.HasPrefix(boundary, "velocity.") || strings.HasPrefix(boundary, "latents.") || strings.HasPrefix(boundary, "schedule.") || boundary == "pixels" || boundary == "initial_latents" || strings.HasPrefix(boundary, "conditioning.") && !strings.HasSuffix(boundary, ".mask") {
			if _, ok := hashes[boundary]; !ok {
				return fmt.Errorf("MOSS generation omitted %s", boundary)
			}
		}
	}
	return nil
}
