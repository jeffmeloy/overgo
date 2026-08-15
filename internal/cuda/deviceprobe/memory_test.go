package deviceprobe

import "testing"

func TestParseMemory(t *testing.T) {
	got, err := parseMemory([]byte("123, 456\n789, 012\n"))
	if err != nil {
		t.Fatal(err)
	}
	if want := (Memory{UsedMiB: 123, FreeMiB: 456}); got != want {
		t.Fatalf("parseMemory() = %+v, want %+v", got, want)
	}
}

func TestParseMemoryRejectsInvalidOutput(t *testing.T) {
	for _, input := range []string{"", "123", "used, 456", "123, free"} {
		if _, err := parseMemory([]byte(input)); err == nil {
			t.Errorf("parseMemory(%q) unexpectedly succeeded", input)
		}
	}
}
