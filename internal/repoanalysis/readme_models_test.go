package repoanalysis

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// readmeModalities are the kinds of data a served model works in, as the README's table of served models names them.
var readmeModalities = []string{
	"text", "vision-language", "video-language", "audio-language", "genomic", "function calling",
	"text-to-image", "text-to-video", "image generation", "video generation", "video editing",
	"image autoencoder", "video autoencoder", "image embedding",
	"speech recognition", "speech synthesis", "time series", "tabular",
}

// readmeMaking are the modalities whose answer is a picture, a clip or speech, which a sample shows.
var readmeMaking = []string{"text-to-image", "text-to-video", "image generation", "video generation", "video editing", "speech synthesis"}

// readmeUnmeasured is the first-output cell of a served model the scorecard has not measured, which has no sample yet.
const readmeUnmeasured = "not yet measured"

// readmeMediaLink matches a link to a picture, a clip or a recording.
var readmeMediaLink = regexp.MustCompile(`\]\(([^)]+\.(?:png|gif|wav))\)`)

// readmeSection returns the lines of the README section a heading opens, up to the next heading of its level or above.
func readmeSection(text, heading string) []string {
	// level is a heading's count of opening marks, and none for a line that is no heading.
	level := func(line string) int {
		marks := len(line) - len(strings.TrimLeft(line, "#"))
		if marks == 0 || !strings.HasPrefix(line[marks:], " ") {
			return 0
		}
		return marks
	}
	var lines []string
	inside := false
	for line := range strings.SplitSeq(text, "\n") {
		if inside && level(line) != 0 && level(line) <= level(heading) {
			break
		}
		if line == heading {
			inside = true
			continue
		}
		if inside {
			lines = append(lines, line)
		}
	}
	return lines
}

// readmeModelFindings reads the README at root and says each way it fails to name a served model with its modality, or
// to show a sample under assets that the repository carries (carried: what git tracks or would track if added).
func readmeModelFindings(root string, carried []string) ([]string, error) {
	data, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		return nil, err
	}
	text := string(data)
	var findings []string
	var header []string
	var rows [][]string
	for _, line := range readmeSection(text, "### Served models") {
		if !strings.HasPrefix(line, "| ") {
			continue
		}
		cells := strings.Split(strings.Trim(line, "|"), "|")
		for index := range cells {
			cells[index] = strings.TrimSpace(cells[index])
		}
		switch {
		case header == nil:
			header = cells
		case strings.Trim(cells[0], "-") != "":
			rows = append(rows, cells)
		}
	}
	model, modality, first := slices.Index(header, "Model"), slices.Index(header, "Modality"), slices.Index(header, "First output")
	if model < 0 || modality < 0 || first < 0 || !slices.Contains(header, "Input → output") || len(rows) == 0 {
		return append(findings, fmt.Sprintf("the served models table has no rows under Model, Modality, Input → output and First output: %q", header)), nil
	}
	// A sample is a paragraph of Sample outputs that opens with the model's name in bold, and every line to the next.
	samples := map[string][]string{}
	var opened string
	section := strings.Join(readmeSection(text, "### Sample outputs"), "\n")
	for line := range strings.SplitSeq(section, "\n") {
		if strings.HasPrefix(line, "**") {
			opened = line
		}
		if opened != "" {
			samples[opened] = append(samples[opened], line)
		}
	}
	seen := map[string]bool{}
	for _, row := range rows {
		if len(row) != len(header) {
			findings = append(findings, fmt.Sprintf("row %q has %d cells under %d columns", row[0], len(row), len(header)))
			continue
		}
		name := row[model]
		if name == "" || seen[name] {
			findings = append(findings, fmt.Sprintf("row %q names no model or one named before", name))
		}
		seen[name] = true
		kinds := strings.Split(row[modality], ", ")
		for _, kind := range kinds {
			if !slices.Contains(readmeModalities, kind) {
				findings = append(findings, fmt.Sprintf("%s: modality %q is none of %q", name, kind, readmeModalities))
			}
		}
		if row[first] == readmeUnmeasured || !slices.ContainsFunc(kinds, func(kind string) bool { return slices.Contains(readmeMaking, kind) }) {
			continue
		}
		shown := false
		for opening, lines := range samples {
			if strings.HasPrefix(opening, "**"+name+",") || strings.HasPrefix(opening, "**"+name+" ") {
				shown = shown || readmeMediaLink.MatchString(strings.Join(lines, "\n"))
			}
		}
		if !shown {
			findings = append(findings, fmt.Sprintf("%s makes %s and the sample outputs show nothing of it", name, row[modality]))
		}
	}
	for _, match := range readmeMediaLink.FindAllStringSubmatch(section, len(section)) {
		target := match[1]
		if path.Dir(target) != assetsRoot {
			findings = append(findings, fmt.Sprintf("sample %s is not under %s", target, assetsRoot))
			continue
		}
		bytes, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(target)))
		if err != nil {
			findings = append(findings, fmt.Sprintf("sample %s: %v", target, err))
			continue
		}
		if digest := sha256.Sum256(bytes); hex.EncodeToString(digest[:]) != contentDigest(target) {
			findings = append(findings, fmt.Sprintf("sample %s is not named by the SHA-256 of its bytes", target))
		}
		if !slices.Contains(carried, target) {
			findings = append(findings, fmt.Sprintf("sample %s is on disk but git ignores it, so a clone lacks it", target))
		}
	}
	return findings, nil
}

// TestReadmeNamesEveryModelAndItsSamples holds the README's served models and its samples to the tree.
//   - the served models table has a Model, a Modality, an Input → output and a First output column
//   - each row names a model once, its modality from readmeModalities
//   - a measured model whose modality makes a picture, a clip or speech has a sample: a paragraph of Sample outputs that
//     opens with its name in bold and links one
//   - every picture, clip and recording the README links is under assets, there, named by the SHA-256 of its bytes, and
//     carried by git: an ignore rule over assets once left a clone without the samples the checkout showed
func TestReadmeNamesEveryModelAndItsSamples(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	carried, err := TrackedDocuments(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	findings, err := readmeModelFindings(root, carried)
	if err != nil {
		t.Fatal(err)
	}
	for _, finding := range findings {
		t.Error(finding)
	}
	picture, ignored := []byte("a picture"), []byte("an ignored picture")
	digest, ignoredDigest := sha256.Sum256(picture), sha256.Sum256(ignored)
	named := assetsRoot + "/" + hex.EncodeToString(digest[:]) + ".png"
	hidden := assetsRoot + "/" + hex.EncodeToString(ignoredDigest[:]) + ".png"
	misnamed := assetsRoot + "/" + strings.Repeat("0", hex.EncodedLen(sha256.Size)) + ".png"
	table := "### Served models\n\n| Model | Modality | Input → output | First output |\n| --- | --- | --- | --- |\n"
	reader := "| Reader | text | text → text | 1 s |\n"
	for _, test := range []struct {
		name, readme, want string
	}{
		{"a sample of each model that makes one", table + "| Painter | text-to-image | text → image | 1 s |\n| Reader | text | text → text | 1 s |\n| Next | text-to-image | text → image | not yet measured |\n\n### Sample outputs\n\n**Painter, text → image.** A fox.\n\n![fox](" + named + ")\n\n## Training\n", ""},
		{"a modality the table does not name", table + "| Painter | painting | text → image | 1 s |\n", `modality "painting"`},
		{"no modality column", "### Served models\n\n| Model | Input → output | First output |\n| --- | --- | --- |\n| Reader | text → text | 1 s |\n", "no rows under"},
		{"a model named twice", table + "| Reader | text | text → text | 1 s |\n| Reader | text | text → text | 1 s |\n", "named before"},
		{"a model that makes pictures and shows none", table + "| Painter | text-to-image | text → image | 1 s |\n\n### Sample outputs\n\n**Reader, text → text.**\n", "show nothing of it"},
		{"a sample outside assets", table + reader + "\n### Sample outputs\n\n![fox](docs/media_samples/fox.png)\n", "is not under assets"},
		{"a sample not named by its bytes", table + reader + "\n### Sample outputs\n\n![fox](" + misnamed + ")\n", "not named by the SHA-256"},
		{"a sample that is not there", table + reader + "\n### Sample outputs\n\n![fox](" + assetsRoot + "/gone.png)\n", "gone.png"},
		{"a sample git ignores", table + reader + "\n### Sample outputs\n\n![fox](" + hidden + ")\n", "git ignores it"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			root := t.TempDir()
			if err := os.MkdirAll(filepath.Join(root, assetsRoot), 0o755); err != nil {
				t.Fatal(err)
			}
			for name, data := range map[string][]byte{"README.md": []byte(test.readme), named: picture, misnamed: picture, hidden: ignored} {
				if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(name)), data, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			// The fixture's git carries its named pictures; the hidden one is on disk under an ignore rule.
			findings, err := readmeModelFindings(root, []string{named, misnamed})
			if err != nil {
				t.Fatal(err)
			}
			joined := strings.Join(findings, "\n")
			if (test.want == "") != (joined == "") || !strings.Contains(joined, test.want) {
				t.Fatalf("want a finding of %q, got %q", test.want, joined)
			}
		})
	}
}
