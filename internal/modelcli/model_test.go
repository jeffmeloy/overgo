package modelcli

import (
	"flag"
	"testing"

	"overgo/internal/dataroot"
)

func TestModelFlagsOpenOptions(t *testing.T) {
	flags := flag.NewFlagSet("test", flag.ContinueOnError)
	modelFlags := AddModelFlags(flags, "adapter")
	if err := flags.Parse([]string{"-device", "2", "-lora", "a.gguf", "-lora", "b.gguf"}); err != nil {
		t.Fatal(err)
	}
	options := modelFlags.OpenOptions(0.5)
	if options.DeviceOrdinal != 2 {
		t.Fatalf("unexpected options: %+v", options)
	}
	if len(options.LoRAAdapters) != 2 || options.LoRAAdapters[0].Scale != 0.5 || options.LoRAAdapters[1].Path != "b.gguf" {
		t.Fatalf("unexpected adapters: %+v", options.LoRAAdapters)
	}
}

func TestModelFlagsRepositoryAndAdapters(t *testing.T) {
	t.Setenv(dataroot.Env, t.TempDir())
	flags := flag.NewFlagSet("test", flag.ContinueOnError)
	modelFlags := AddModelFlags(flags, "adapter")
	// Preserve the reviewed CLI default: CUDA ordinals start at zero.
	if *modelFlags.DeviceOrdinal != 0 {
		t.Fatalf("default device ordinal = %d, want 0", *modelFlags.DeviceOrdinal)
	}
	if _, err := modelFlags.RepositoryPath(); err == nil {
		t.Fatal("missing default repository accepted")
	}
	if err := flags.Parse([]string{"-repo", "chosen-store", "-lora", "adapter.gguf"}); err != nil {
		t.Fatal(err)
	}
	if repository, err := modelFlags.RepositoryPath(); err != nil || repository != "chosen-store" {
		t.Fatalf("explicit repository = %q, %v", repository, err)
	}
	paths := modelFlags.LoRAPaths()
	paths[0] = "changed.gguf"
	if got := modelFlags.LoRAPaths(); len(got) != 1 || got[0] != "adapter.gguf" {
		t.Fatalf("adapter paths alias flag storage: %v", got)
	}
	var absent *ModelFlags
	if _, err := absent.RepositoryPath(); err == nil {
		t.Fatal("absent flags accepted")
	}
	if _, err := OpenRunner(t.Context(), "", "missing.gguf", BuildOpenOptions(0, nil, 1)); err == nil {
		t.Fatal("model opening accepted an empty repository")
	}
}

func TestModelFlagsRejectEmptyLoRA(t *testing.T) {
	flags := flag.NewFlagSet("test", flag.ContinueOnError)
	AddModelFlags(flags, "adapter")
	if err := flags.Parse([]string{"-lora", "  "}); err == nil {
		t.Fatal("expected empty LoRA path rejection")
	}
}
