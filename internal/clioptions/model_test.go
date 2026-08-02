package clioptions

import (
	"flag"
	"testing"
)

func TestModelFlagsOpenOptions(t *testing.T) {
	flags := flag.NewFlagSet("test", flag.ContinueOnError)
	modelFlags := AddModelFlags(flags, "adapter")
	if err := flags.Parse([]string{"-device", "2", "-preload", "-native-q8", "-lora", "a.gguf", "-lora", "b.gguf"}); err != nil {
		t.Fatal(err)
	}
	options := modelFlags.OpenOptions(0.5)
	if options.DeviceOrdinal != 2 || !options.PreloadDeviceWeights || !options.PreloadQuantizedWeights {
		t.Fatalf("unexpected options: %+v", options)
	}
	if len(options.LoRAAdapters) != 2 || options.LoRAAdapters[0].Scale != 0.5 || options.LoRAAdapters[1].Path != "b.gguf" {
		t.Fatalf("unexpected adapters: %+v", options.LoRAAdapters)
	}
}

func TestModelFlagsRejectEmptyLoRA(t *testing.T) {
	flags := flag.NewFlagSet("test", flag.ContinueOnError)
	AddModelFlags(flags, "adapter")
	if err := flags.Parse([]string{"-lora", "  "}); err == nil {
		t.Fatal("expected empty LoRA path rejection")
	}
}
