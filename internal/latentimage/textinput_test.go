package latentimage

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/hfbpe"
)

// leInt64SHA hashes a sequence of ints as concatenated little-endian int64 -- the
// exact-contract encoding used to compare token-id / mask vectors byte-for-byte.
func leInt64SHA(vals []int) string {
	h := sha256.New()
	var b [8]byte
	for _, v := range vals {
		binary.LittleEndian.PutUint64(b[:], uint64(int64(v)))
		h.Write(b[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func maskInts(m []bool) []int {
	out := make([]int, len(m))
	for i, v := range m {
		if v {
			out[i] = 1
		}
	}
	return out
}

// The Krea golden fox case (dtc brick 1/3). Fixed by the cited template facts +
// the real Qwen2 tokenizer; seed is irrelevant to text input.
const kreaGoldenPrompt = "a red fox licking a vanilla ice cream cone in snow"

// Golden SHAs of the rendered ids + mask (LE-int64 encoding). Pinned from the real
// Krea-2-Turbo tokenizer; ORACLE: adaptive prepareSelectedLayerTextInput reproduced
// from the cited encoder.py template facts + hfbpe (a verbatim port of adaptive's
// tokenizer over the same tokenizer.json). The 34/5 prefix/suffix anchors and these
// SHAs together detect any tokenizer or template drift.
const (
	kreaGoldenIDsSHA  = "b0fe1c0b43f92fc37c3d804791970669f6ac098c60e7ba12f6c82e2b8d527560"
	kreaGoldenMaskSHA = "be1a811db02b1ffb5f5bff0fec67336803eb66c5cbca199e23bea4c1f6799c63"
)

// TestRenderKreaTextInputLayout exercises the pure fixed-row layout math (no
// tokenizer, runs in CI): [prefix][prompt][pad...][suffix] with the pad region
// unattended and PromptRows == maxPromptTokens.
func TestRenderKreaTextInputLayout(t *testing.T) {
	prefix := []int{100, 101, 102} // 3 rows
	prompt := []int{200, 201}      // 2 rows
	suffix := []int{300}           // 1 row
	const maxPrompt = 5
	const padID = 9
	in := assembleTextRows(prefix, prompt, suffix, maxPrompt, padID)

	// total = maxPrompt + len(prefix) = 5 + 3 = 8; padded = 5+3-1 = 7.
	wantIDs := []int{100, 101, 102, 200, 201, padID, padID, 300}
	wantMask := []bool{true, true, true, true, true, false, false, true}
	if in.PromptRows != maxPrompt {
		t.Fatalf("PromptRows=%d want %d", in.PromptRows, maxPrompt)
	}
	if len(in.IDs) != len(wantIDs) || len(in.Mask) != len(wantMask) {
		t.Fatalf("len ids=%d mask=%d want %d", len(in.IDs), len(in.Mask), len(wantIDs))
	}
	for i := range wantIDs {
		if in.IDs[i] != wantIDs[i] || in.Mask[i] != wantMask[i] {
			t.Fatalf("row %d: id=%d mask=%v want id=%d mask=%v", i, in.IDs[i], in.Mask[i], wantIDs[i], wantMask[i])
		}
	}

	// Over-long prompt is truncated to maxPrompt (mirrors adaptive).
	long := []int{200, 201, 202, 203, 204, 205, 206}
	tr := assembleTextRows(prefix, long, suffix, maxPrompt, padID)
	if len(tr.IDs) != maxPrompt+len(prefix) || tr.PromptRows != maxPrompt {
		t.Fatalf("truncated: ids=%d promptRows=%d", len(tr.IDs), tr.PromptRows)
	}
	// prompt fills the whole maxPrompt budget after prefix, no pad rows remain.
	for i := len(prefix); i < len(prefix)+maxPrompt; i++ {
		if !tr.Mask[i] {
			t.Fatalf("truncated row %d unexpectedly unattended", i)
		}
	}
}

// TestRenderKreaTextInputGoldenSHA renders the fox prompt with the real Krea-2-Turbo
// Qwen2 tokenizer and asserts the exact ids + mask (SHA of the LE-int64 encoding)
// against the pinned golden, plus the template-boundary anchors (34 prefix / 5
// suffix tokens per encoder.py) and the pad/mask layout. Gated on the model dir
// because the tokenizer.json is required.
func TestRenderKreaTextInputGoldenSHA(t *testing.T) {
	dir := kreaDirOrSkip(t)
	tok, err := hfbpe.Load(filepath.Join(dir, "tokenizer"))
	if err != nil {
		t.Fatalf("load Qwen2 tokenizer: %v", err)
	}
	tmpl := kreaProfileOrSkip(t).Prompt

	// Live-citation checks: the cited pad token + template counts vs the artifact.
	assertKreaTemplateCitations(t, dir, tok, tmpl)

	in, err := renderTextInput(tok, kreaGoldenPrompt, tmpl)
	if err != nil {
		t.Fatalf("RenderKreaTextInput: %v", err)
	}

	// Row geometry: total = MaxPromptTokens + len(prefix)=34; PromptRows=512.
	wantTotal := tmpl.MaxTokens + tmpl.PrefixTokens // 512 + 34 = 546
	if len(in.IDs) != wantTotal || len(in.Mask) != wantTotal {
		t.Fatalf("ids=%d mask=%d want %d", len(in.IDs), len(in.Mask), wantTotal)
	}
	if in.PromptRows != tmpl.MaxTokens {
		t.Fatalf("PromptRows=%d want %d", in.PromptRows, tmpl.MaxTokens)
	}

	// Layout invariants: prefix rows attended; a pad gap (padID, unattended)
	// precedes the 5 attended suffix rows at [padded:].
	padID, _ := tok.SpecialID(tmpl.PadToken)
	padded := tmpl.MaxTokens + tmpl.PrefixTokens - tmpl.SuffixTokens // 541
	trueMask := 0
	padRows := 0
	for i, id := range in.IDs {
		if in.Mask[i] {
			trueMask++
		}
		if !in.Mask[i] {
			if id != padID {
				t.Fatalf("unattended row %d id=%d != pad %d", i, id, padID)
			}
			padRows++
		}
	}
	for i := padded; i < len(in.IDs); i++ {
		if !in.Mask[i] {
			t.Fatalf("suffix row %d unattended", i)
		}
	}
	// prompt token count P = attended - prefix(34) - suffix(5); pad rows = 507 - P.
	promptLen := trueMask - tmpl.PrefixTokens - tmpl.SuffixTokens
	if padRows != tmpl.MaxTokens-tmpl.SuffixTokens-promptLen {
		t.Fatalf("pad rows=%d inconsistent with prompt len=%d", padRows, promptLen)
	}

	idsSHA := leInt64SHA(in.IDs)
	maskSHA := leInt64SHA(maskInts(in.Mask))
	t.Logf("krea fox golden: total=%d promptRows=%d attended=%d padRows=%d promptTokens=%d",
		len(in.IDs), in.PromptRows, trueMask, padRows, promptLen)
	t.Logf("krea fox ids  LE-int64 sha256 = %s", idsSHA)
	t.Logf("krea fox mask LE-int64 sha256 = %s", maskSHA)

	if kreaGoldenIDsSHA != "PLACEHOLDER_IDS" {
		if idsSHA != kreaGoldenIDsSHA {
			t.Fatalf("ids sha mismatch:\n got %s\nwant %s", idsSHA, kreaGoldenIDsSHA)
		}
		if maskSHA != kreaGoldenMaskSHA {
			t.Fatalf("mask sha mismatch:\n got %s\nwant %s", maskSHA, kreaGoldenMaskSHA)
		}
	}
}

// assertKreaTemplateCitations proves the cited pad-token fact is live against the
// artifact's tokenizer_config.json, and the prefix/suffix tokenize to the reference
// conditioner's asserted counts (encoder.py prompt_template_encode_start_idx=34 /
// suffix_start_idx=5) -- so a stale citation or tokenizer drift fails loudly.
func assertKreaTemplateCitations(t *testing.T, dir string, tok *hfbpe.Tokenizer, tmpl textTemplate) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer", "tokenizer_config.json"))
	if err != nil {
		t.Fatalf("read tokenizer_config.json: %v", err)
	}
	var cfg struct {
		PadToken string `json:"pad_token"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("parse tokenizer_config.json: %v", err)
	}
	if cfg.PadToken != tmpl.PadToken {
		t.Fatalf("cited pad_token %q != artifact %q (stale citation)", tmpl.PadToken, cfg.PadToken)
	}
	if _, ok := tok.SpecialID(tmpl.PadToken); !ok {
		t.Fatalf("pad token %q not resolvable in tokenizer", tmpl.PadToken)
	}
	pre, err := tok.Encode(tmpl.Prefix)
	if err != nil {
		t.Fatalf("encode prefix: %v", err)
	}
	suf, err := tok.Encode(tmpl.Suffix)
	if err != nil {
		t.Fatalf("encode suffix: %v", err)
	}
	if len(pre) != tmpl.PrefixTokens {
		t.Fatalf("prefix tokenized to %d ids, encoder.py asserts %d", len(pre), tmpl.PrefixTokens)
	}
	if len(suf) != tmpl.SuffixTokens {
		t.Fatalf("suffix tokenized to %d ids, encoder.py asserts %d", len(suf), tmpl.SuffixTokens)
	}
}
