//go:build modeltest

package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheUltraRecordingsAgreeAndParseStrictly holds MOSS Ultra's ComfyUI recordings to what they claim
// (comfy-recorder/record): each section of the fixture's workflow recorded twice at 1024 x 576 and text to image once
// more at the workflow's own size, each recording parsing strictly, naming the workflow the fixture keeps and every
// file it kept, which lies beside the store under references/comfy/moss-ultra by the SHA-256 it names; and a section's
// second recording gives every node's output the bytes its first gave, the pictures, the sharpened pictures, the depth
// maps and the final latents alike. ComfyUI numbers what it saves, so a node's files are matched by node and kind.
func TestTheUltraRecordingsAgreeAndParseStrictly(t *testing.T) {
	root := testutil.RepoRoot(t)
	ultra := testcheck.Must(readFixture(filepath.Join(root, ultraFixture)))(t)
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	references := filepath.Join(roots.References(), "comfy", "moss-ultra")
	var readFiles []string
	read := func(run, section string) recorded {
		t.Helper()
		var taken recorded
		name := filepath.ToSlash(filepath.Join(ultraFixture, "recordings", run, slug(section)+".json"))
		readFiles = append(readFiles, name)
		data := testcheck.Must(os.ReadFile(filepath.Join(root, name)))(t)
		testcheck.NoError(t, strictjson.DecodeBytes(data, &taken))
		// The recorder's own type reads it strictly too: a field it does not write is refused.
		var written recording
		testcheck.NoError(t, strictjson.DecodeBytes(data, &written))
		if taken.Section != section || taken.Workflow != hashOf(ultra.workflow) || taken.Prompt == "" || taken.Milliseconds <= 0 || taken.PeakCardBytes == 0 {
			t.Fatalf("%s %s names section %q, workflow %s, prompt %q, %d ms, %d bytes; want this section of the kept workflow %s, its prompt, time and memory",
				run, section, taken.Section, taken.Workflow, taken.Prompt, taken.Milliseconds, taken.PeakCardBytes, hashOf(ultra.workflow))
		}
		for _, file := range taken.Files {
			data, err := os.ReadFile(filepath.Join(references, run, slug(section), file.Name))
			if err != nil || hashOf(data) != file.SHA256 || len(data) != file.Bytes {
				t.Errorf("%s %s keeps %s as %s of %d bytes; read %d bytes (%v)", run, section, file.Name, file.SHA256, file.Bytes, len(data), err)
			}
		}
		return taken
	}
	// outputs are a recording's saved files by node and kind, each node's in the order ComfyUI saved them. A preview its
	// page shows (type temp) is kept and read by its hash, but not held to agree: ComfyUI answers a node from an earlier
	// prompt's cache where its inputs match, so the turbo depth section's depth preview was the plain section's on the
	// second run and its own on the first.
	outputs := func(taken recorded) map[string][]string {
		byNode := map[string][]string{}
		for _, file := range taken.Files {
			if file.Type == "output" {
				byNode[file.Node+" "+file.Kind] = append(byNode[file.Node+" "+file.Kind], file.SHA256)
			}
		}
		return byNode
	}
	if titles := ultra.titles(); len(titles) != 6 {
		t.Fatalf("the workflow has sections %v; want its six", titles)
	}
	for _, section := range ultra.titles() {
		first, second := read("run-1", section), read("run-2", section)
		if first.Request != second.Request || !slices.Equal(sortedCommits(first), sortedCommits(second)) {
			t.Errorf("%s: the two recordings posted %s and %s at commits %v and %v; want one request at one install", section,
				first.Request, second.Request, first.Commits, second.Commits)
		}
		if first.Set[ultraSizeInput(first)] != 0.5625 {
			t.Errorf("%s: recorded with inputs %v; want its ResolutionSelector at 0.5625 megapixels, 1024 x 576", section, first.Set)
		}
		if !strings.Contains(strings.Join(slices.Collect(func(yield func(string) bool) {
			for _, file := range first.Files {
				if !yield(file.Kind) {
					return
				}
			}
		}), " "), "latents") {
			t.Errorf("%s: the recording keeps no latent", section)
		}
		want, got := outputs(first), outputs(second)
		if len(want) == 0 {
			t.Errorf("%s: the recording keeps no output, only previews", section)
		}
		for node, hashes := range want {
			if !slices.Equal(got[node], hashes) {
				t.Errorf("%s: node %s gave %v the second time; want the bytes of the first, %v", section, node, got[node], hashes)
			}
		}
	}
	speed := read("speed", "TEXT TO IMAGE")
	if slices.Sort(readFiles); !slices.Equal(readFiles, recordingFiles) {
		t.Errorf("the test read %v; want the recordings it names, %v", readFiles, recordingFiles)
	}
	if len(speed.Set) != 0 {
		t.Errorf("the speed recording set %v; want the workflow's own size, nothing set", speed.Set)
	}
}

// recordingFiles are the recordings the test reads, each named whole, as every fixture is by its reader.
var recordingFiles = []string{
	"fixtures/comfy/moss-ultra/recordings/run-1/depth-control-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/run-1/depth-control.json",
	"fixtures/comfy/moss-ultra/recordings/run-1/image-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/run-1/image-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/run-1/text-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/run-1/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/run-2/depth-control-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/run-2/depth-control.json",
	"fixtures/comfy/moss-ultra/recordings/run-2/image-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/run-2/image-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/run-2/text-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/run-2/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/speed/text-to-image.json",
}

// sortedCommits are a recording's commits as folder=commit lines, sorted.
func sortedCommits(taken recorded) []string {
	var lines []string
	for folder, commit := range taken.Commits {
		lines = append(lines, folder+"="+commit)
	}
	slices.Sort(lines)
	return lines
}

// ultraSizeInput is the input a 1024 x 576 recording set over its section's own: its ResolutionSelector's megapixels.
func ultraSizeInput(taken recorded) string {
	for input := range taken.Set {
		if strings.HasSuffix(input, ".megapixels") {
			return input
		}
	}
	return ""
}
