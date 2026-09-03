package inference

import "testing"

// Two pasts that append into the same target capacity can sit on pages of
// different capacity (256 tokens on a 256-token page and 300 on a 512-token
// page both append into 512); a decode session compiled over the larger
// page must not serve the smaller one, because its cache-append copy reads
// the compiled page extent.
func TestDecodeSessionIdentityDistinguishesSourcePageCapacity(t *testing.T) {
	var lora [32]byte
	compiled := decodeSessionIdentity{
		capacity: 512, source: 512, branches: 1, tokenCount: 1, output: deviceOutputPlan{}, lora: lora,
	}
	if !compiled.matches(512, 512, 1, deviceOutputPlan{}, lora) {
		t.Fatal("the identity does not match itself")
	}
	if compiled.matches(512, 256, 1, deviceOutputPlan{}, lora) {
		t.Fatal("a session compiled over a 512-token page matched a 256-token past page")
	}
	if compiled.matches(1024, 512, 1, deviceOutputPlan{}, lora) {
		t.Fatal("a different target capacity matched")
	}
	if page := cachePageCapacity(256, 4096, "capacity"); page != 256 {
		t.Fatalf("256 tokens page to %d, want 256", page)
	}
	if page := cachePageCapacity(300, 4096, "capacity"); page != 512 {
		t.Fatalf("300 tokens page to %d, want 512", page)
	}
	if next := cachePageCapacity(257, 4096, "capacity"); next != 512 {
		t.Fatalf("257 tokens append into %d, want 512", next)
	}
}
