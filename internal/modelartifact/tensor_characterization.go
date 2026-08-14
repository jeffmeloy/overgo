package modelartifact

import (
	"errors"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/tensorstats"
)

// TensorCharacterization is one tensor's storage identity plus its
// distribution-free value profile, for the analysis workbench. The embedded
// profile fields flatten into the JSON object.
type TensorCharacterization struct {
	Name    string   `json:"name"`
	Storage string   `json:"storage"`
	Shape   []uint64 `json:"shape"`
	tensorstats.Characterization
}

// CharacterizeGGUFTensors samples every tensor in an open GGUF and returns a
// distribution-free characterization per tensor, bounded by policy. It reads
// only sampled blocks and never buffers a whole tensor. Non-finite values are
// excluded and reported through FiniteSamples; a tensor with no finite sampled
// value is returned with its counts but an otherwise-zero profile rather than
// failing the whole request.
func CharacterizeGGUFTensors(file *gguf.File, policy MeasurementPolicy) ([]TensorCharacterization, error) {
	if file == nil {
		return nil, errors.New("model artifact: nil GGUF characterization source")
	}
	out := make([]TensorCharacterization, 0, len(file.Tensors))
	var readBytes uint64
	for _, tensor := range file.Tensors {
		samples, elements, bytesNeeded, err := sampleGGUFTensorValues(file, tensor, policy.MaxSamplesPerTensor)
		if err != nil {
			return nil, err
		}
		if readBytes > policy.MaxReadBytes || bytesNeeded > policy.MaxReadBytes-readBytes {
			return nil, errors.New("model artifact: GGUF characterization exceeds read budget")
		}
		readBytes += bytesNeeded
		entry := TensorCharacterization{
			Name:    tensor.Name,
			Storage: strings.ToLower(tensor.Type.String()),
			Shape:   append([]uint64(nil), tensor.Shape[:tensor.Dimensions]...),
		}
		if profile, ok := tensorstats.Characterize(elements, samples); ok {
			entry.Characterization = profile
		} else {
			entry.Characterization = tensorstats.Characterization{Elements: elements, Samples: uint64(len(samples))}
		}
		out = append(out, entry)
	}
	return out, nil
}
