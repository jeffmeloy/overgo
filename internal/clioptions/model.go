package clioptions

import (
	"errors"
	"flag"
	"slices"
	"strings"

	"llamacpp2go/internal/inference"
)

// ModelFlags: common model-loading flags.
type ModelFlags struct {
	DeviceOrdinal *int
	Preload       *bool
	NativeQ8      *bool
	NativeQuant   *bool
	HostCache     *bool
	loraPaths     stringList
}

// ModelFlagConfig: flag names and defaults
type ModelFlagConfig struct {
	PreloadName        string
	PreloadDefault     bool
	NativeQuantName    string
	NativeQuantDefault bool
	NativeQ8Name       string
	NativeQ8Default    bool
	HostCacheName      string
	HostCacheDefault   bool
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
	return AddModelFlagsWithConfig(flags, loraHelp, ModelFlagConfig{
		PreloadName: "preload", NativeQuantName: "native-quant", NativeQ8Name: "native-q8",
		HostCacheName: "host-cache",
	})
}

// AddModelFlagsWithConfig: configured model flag registration
func AddModelFlagsWithConfig(flags *flag.FlagSet, loraHelp string, config ModelFlagConfig) *ModelFlags {
	result := &ModelFlags{}
	result.DeviceOrdinal = flags.Int("device", 0, "CUDA device ordinal")
	if config.PreloadName != "" {
		result.Preload = flags.Bool(config.PreloadName, config.PreloadDefault, "dequantize all model weights once into CUDA memory")
	}
	if config.NativeQ8Name != "" {
		result.NativeQ8 = flags.Bool(config.NativeQ8Name, config.NativeQ8Default, "preload Q8_0 weights without dequantizing them")
	}
	if config.NativeQuantName != "" {
		result.NativeQuant = flags.Bool(config.NativeQuantName, config.NativeQuantDefault, "preload supported quantized weights without dequantizing them")
	}
	if config.HostCacheName != "" {
		result.HostCache = flags.Bool(
			config.HostCacheName,
			config.HostCacheDefault,
			"retain lazily dequantized F32 weights in host memory",
		)
	}
	flags.Var(&result.loraPaths, "lora", loraHelp)
	return result
}

// OpenOptions: inference model-loading options.
func (flags *ModelFlags) OpenOptions(loraScale float32) inference.OpenOptions {
	preload := flags.Preload != nil && *flags.Preload
	native := flags.NativeQ8 != nil && *flags.NativeQ8 || flags.NativeQuant != nil && *flags.NativeQuant
	result := BuildOpenOptions(*flags.DeviceOrdinal, preload, native, flags.loraPaths, loraScale)
	result.CacheHostWeights = flags.HostCache != nil && *flags.HostCache
	return result
}

// LoRAPaths: copied adapter paths
func (flags *ModelFlags) LoRAPaths() []string {
	return slices.Clone(flags.loraPaths)
}

// BuildOpenOptions: common inference open options
func BuildOpenOptions(device int, preload, nativeQuant bool, loraPaths []string, loraScale float32) inference.OpenOptions {
	adapters := make([]inference.LoRAConfig, len(loraPaths))
	for index, path := range loraPaths {
		adapters[index] = inference.LoRAConfig{Path: path, Scale: loraScale}
	}
	return inference.OpenOptions{
		DeviceOrdinal: device, PreloadDeviceWeights: preload,
		PreloadQuantizedWeights: nativeQuant, LoRAAdapters: adapters,
	}
}
