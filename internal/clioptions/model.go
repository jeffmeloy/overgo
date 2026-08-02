package clioptions

import (
	"errors"
	"flag"
	"strings"

	"llamacpp2go/internal/inference"
)

// ModelFlags: common model-loading flags.
type ModelFlags struct {
	DeviceOrdinal *int
	Preload       *bool
	NativeQ8      *bool
	NativeQuant   *bool
	loraPaths     stringList
}

type stringList []string

func (values *stringList) String() string {
	return strings.Join(*values, ",")
}

func (values *stringList) Set(value string) error {
	if strings.TrimSpace(value) == "" {
		return errors.New("LoRA path is empty")
	}
	*values = append(*values, value)
	return nil
}

// AddModelFlags: common model-loading flag registration.
func AddModelFlags(flags *flag.FlagSet, loraHelp string) *ModelFlags {
	result := &ModelFlags{}
	result.DeviceOrdinal = flags.Int("device", 0, "CUDA device ordinal")
	result.Preload = flags.Bool("preload", false, "dequantize all model weights once into CUDA memory")
	result.NativeQ8 = flags.Bool("native-q8", false, "preload Q8_0 weights without dequantizing them")
	result.NativeQuant = flags.Bool("native-quant", false, "preload supported quantized weights without dequantizing them")
	flags.Var(&result.loraPaths, "lora", loraHelp)
	return result
}

// OpenOptions: inference model-loading options.
func (flags *ModelFlags) OpenOptions(loraScale float32) inference.OpenOptions {
	adapters := make([]inference.LoRAConfig, len(flags.loraPaths))
	for index, path := range flags.loraPaths {
		adapters[index] = inference.LoRAConfig{Path: path, Scale: loraScale}
	}
	return inference.OpenOptions{
		DeviceOrdinal:           *flags.DeviceOrdinal,
		PreloadDeviceWeights:    *flags.Preload,
		PreloadQuantizedWeights: *flags.NativeQ8 || *flags.NativeQuant,
		LoRAAdapters:            adapters,
	}
}
