package repoanalysis

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"overgo/internal/gosyntax"
)

// TestDocsHoldNothingSupersededOrDangling holds the documents to what is read
// and current. A media sample is named by the digest of its bytes, and it is
// read only by what carries that digest whole: a read document, as an output
// identity, a path or a key, or a test, as the digest or the file name. The
// directory the exporter writes into reads none of them, so a sample that its
// last record stops naming is refused in the commit that leaves it behind; a
// digest inside a longer run of hexadecimal digits is another number; the plan
// keeps nothing; and a document nothing reads keeps nothing either. A file
// that is not named by a digest is read as before, by the directory a tool
// walks. A record of a run is never refused for what it names: one that names
// a command, a fixture or a test the tree no longer has is history, and the
// scan that says so reports exactly those names. On the checkout itself the
// test measures the samples, the pages and the records, and logs how many
// records are history.
func TestDocsHoldNothingSupersededOrDangling(t *testing.T) {
	t.Parallel()
	// Invented digests, one repeated digit each: a real sample's digest in this
	// file would make this package that sample's reader.
	digest := func(digit rune) string { return strings.Repeat(string(digit), hex.EncodedLen(sha256.Size)) }
	identity, located, keyed, worded, filed := digest('a'), digest('b'), digest('7'), digest('d'), digest('e')
	unnamed, planned, embedded, walked, strayed := digest('1'), digest('2'), digest('f'), digest('4'), digest('5')
	// As long as a digest, and no number: a name, read as any other.
	lettered := digest('z')
	samples := "docs/media_samples/"

	t.Run("a sample is read by what carries its digest and by nothing else", func(t *testing.T) {
		snapshot, err := gosyntax.Snapshot{}.Overlay(map[string][]byte{
			"cmd/export/samples.go": []byte("package main\n" +
				"var sampleRoot, pictureRoot = \"docs/media_samples\", \"docs/pictures\"\n" +
				"var read = []string{\"docs/verification/coverage.json\", \"docs/validation.json\", \"docs/processing.json\", \"docs/plan.json\"}\n"),
			"internal/frozen/frozen_test.go": []byte("package frozen\n" +
				"var want = map[string]string{\"" + worded + "\": \"" + filed + ".gif\"}\n"),
		})
		if err != nil {
			t.Fatal(err)
		}
		texts := map[string]string{
			"docs/verification/coverage.json": `{"bound": "output:sha256:` + identity + `", "left": "A` + embedded + `", "right": "` + embedded + `0", "ran": "cmd/gone TestGone"}`,
			"docs/validation.json":            `{"sample": "` + samples + located + `.gif"}`,
			"docs/processing.json":            `{"` + keyed + `.gif": {"frames": 8}}`,
			"docs/plan.json":                  `{"rationale": "remove ` + samples + planned + `.gif"}`,
			"docs/notes.txt":                  "kept for " + strayed,
			samples + identity + ".png":       "",
			samples + located + ".gif":        "",
			samples + keyed + ".gif":          "",
			samples + worded + ".png":         "",
			samples + filed + ".gif":          "",
			samples + unnamed + ".wav":        "",
			samples + planned + ".gif":        "",
			samples + embedded + ".png":       "",
			samples + walked + ".png":         "",
			samples + strayed + ".png":        "",
			// Not named by a digest: a plain name, a short word of hexadecimal digits, and the long name that is no number.
			"docs/pictures/plain.png":            "",
			"docs/pictures/decade.png":           "",
			"docs/pictures/" + lettered + ".png": "",
		}
		documents := make([]string, 0, len(texts))
		for document := range texts {
			documents = append(documents, document)
		}
		slices.Sort(documents)
		census, err := TrackedDocumentCensus(snapshot, documents)
		if err != nil {
			t.Fatal(err)
		}
		if err := DocumentReferences(census, func(document string) ([]byte, error) { return []byte(texts[document]), nil }); err != nil {
			t.Fatal(err)
		}
		var refused []string
		for _, finding := range unwrapFindings(ValidateTrackedDocuments(census)) {
			document, _, _ := strings.Cut(strings.TrimPrefix(finding.Error(), "tracked document "), " ")
			refused = append(refused, document)
		}
		slices.Sort(refused)
		want := []string{
			"docs/notes.txt",
			samples + unnamed + ".wav",
			samples + planned + ".gif",
			samples + embedded + ".png",
			samples + walked + ".png",
			samples + strayed + ".png",
		}
		slices.Sort(want)
		if !slices.Equal(refused, want) {
			t.Fatalf("the census refuses\n%s\nwant exactly the samples nothing read carries, and the document nothing reads:\n%s", strings.Join(refused, "\n"), strings.Join(want, "\n"))
		}
	})

	t.Run("a record that names what the tree no longer has is history", func(t *testing.T) {
		root := t.TempDir()
		for name, content := range map[string]string{
			"cmd/tool/main.go":            "package main\n",
			"internal/kept/kept_test.go":  "package kept\n\nimport \"testing\"\n\nfunc TestKept(t *testing.T) {}\n",
			"fixtures/kept/prompts.jsonl": "{}\n",
		} {
			file := filepath.Join(root, filepath.FromSlash(name))
			if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(file, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		record := "docs/verification/probe-of-a-removed-tool.txt"
		texts := map[string]string{
			record:                            "command: go run ./cmd/gone -prompts fixtures/gone.json\nheld by: TestGone, then TestKept\nrerun with: cmd/tool on fixtures/kept/prompts.jsonl\n",
			"docs/verification/a-current.txt": "command: go run ./cmd/tool\nheld by: TestKept\n",
		}
		got := DeadReferences(root, texts)
		want := []string{record + ": TestGone", record + ": cmd/gone", record + ": fixtures/gone.json"}
		if !slices.Equal(got, want) {
			t.Fatalf("the record's absent names are\n%s\nwant the removed command, its removed fixture and the removed test, and nothing the tree still holds:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
		}
	})

	t.Run("the checkout's samples, pages and records are measured", func(t *testing.T) {
		root := "../.."
		documents, err := TrackedDocuments(t.Context(), root)
		if err != nil {
			t.Fatal(err)
		}
		texts, held, records, pages := map[string]string{}, 0, 0, 0
		for _, document := range documents {
			record, page := strings.HasPrefix(document, "docs/verification/"), path.Dir(document) == documentsRoot && path.Ext(document) == ".md"
			switch {
			case strings.HasPrefix(document, samples) && contentDigest(document) != "":
				held++
				continue
			case record:
				records++
			case page:
				pages++
			default:
				continue
			}
			data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(document)))
			if err != nil {
				t.Fatal(err)
			}
			// A record that is a picture names nothing.
			if utf8.Valid(data) {
				texts[document] = string(data)
			}
		}
		history, absent, stalePages := map[string]bool{}, map[string]bool{}, 0
		for _, line := range DeadReferences(root, texts) {
			document, name, _ := strings.Cut(line, ": ")
			if path.Dir(document) == documentsRoot {
				stalePages++
				t.Logf("page %s names %s, which the tree no longer has", document, name)
				continue
			}
			history[document], absent[name] = true, true
		}
		t.Logf("samples=%d records=%d pages=%d; records that are history=%d, naming %d distinct absent paths and tests; absent names in pages=%d",
			held, records, pages, len(history), len(absent), stalePages)
	})
}
