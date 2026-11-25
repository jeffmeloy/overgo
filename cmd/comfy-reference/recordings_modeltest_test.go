//go:build modeltest

package main

import (
	"cmp"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/modelartifact"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheUltraRecordingsAgreeAndParseStrictly holds MOSS Ultra's ComfyUI recordings to what they claim
// (comfy-recorder/record, comfy-recorder/record-rest): each section of the fixture's workflow recorded twice at
// 1024 x 576, text to image once more at the workflow's own size and once more in each of prompt modes 2 and 3, each
// recording parsing strictly, naming the workflow the fixture keeps and every file it kept, which lies beside the store
// under references/comfy/moss-ultra by the SHA-256 it names, each model file it loaded by the SHA-256 the import recipe
// pins, its sampler's sigmas, its steps' count and the last zero, falling, and each of its sampler's steps once, in
// order, as ComfyUI's websocket told it; and a section's second recording gives every node's output the bytes its first
// gave, the pictures, the sharpened pictures, the depth maps and the final latents alike, and the same sigmas. ComfyUI
// numbers what it saves, so a node's files are matched by node and kind.
func TestTheUltraRecordingsAgreeAndParseStrictly(t *testing.T) {
	root := testutil.RepoRoot(t)
	ultra := testcheck.Must(readFixture(filepath.Join(root, ultraFixture)))(t)
	roots := testcheck.Must(dataroot.Resolve(root))(t)
	references := filepath.Join(roots.References(), "comfy", "moss-ultra")
	recipe := testcheck.Must(modelartifact.ReadLocalImport(filepath.Join(root, "internal", "modelrecipe", "moss_ultra_import.json")))(t)
	pins := map[string]string{}
	for _, file := range recipe.Files {
		pins[filepath.Base(file.Path)] = file.SHA256
	}
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
		if taken.Section != section || taken.Workflow != hashOf(ultra.workflow) || taken.Prompt == "" || taken.Milliseconds <= 0 || taken.PeakCardBytes == 0 || taken.Resumed {
			t.Fatalf("%s %s names section %q, workflow %s, prompt %q, %d ms, %d bytes, resumed %t; want this section of the kept workflow %s, its prompt, time and memory, heard whole",
				run, section, taken.Section, taken.Workflow, taken.Prompt, taken.Milliseconds, taken.PeakCardBytes, taken.Resumed, hashOf(ultra.workflow))
		}
		folders := map[string]bool{}
		for _, model := range taken.Models {
			folders[model.Folder] = true
			if model.SHA256 != pins[model.Name] {
				t.Errorf("%s %s names %+v; want the SHA-256 the recipe pins, %s", run, section, model, pins[model.Name])
			}
		}
		if !folders["diffusion_models"] || !folders["text_encoders"] || !folders["vae"] {
			t.Errorf("%s %s names the models %+v; want its diffusion model, encoder and VAE at least", run, section, taken.Models)
		}
		request := testcheck.Must(ultra.request(section))(t)
		for _, id := range slices.Sorted(maps.Keys(request)) {
			if request[id].ClassType != sampler {
				continue
			}
			sigmas, steps := taken.Sigmas[id], len(taken.Sigmas[id])-1
			if given, literal := request[id].Inputs["steps"].(float64); literal && int(given) != steps || steps < 1 || sigmas[steps] != 0 ||
				!slices.IsSortedFunc(sigmas, func(a, b float64) int { return cmp.Compare(b, a) }) {
				t.Errorf("%s %s keeps the sigmas %v of sampler %s; want its steps' count and one more, falling to 0", run, section, sigmas, id)
			}
			// ComfyUI sends a sampler's first and last step, and between them at most one step in 100 ms
			// (comfy/utils.py ProgressBar, PROGRESS_THROTTLE_MIN_INTERVAL), so a step faster than that is told within
			// the span of the event after it: each event is the steps done so far, more than the one before.
			progress := taken.Progress[id]
			for index, stamp := range progress {
				if stamp.Max != steps || index == 0 && stamp.Value != 1 || index > 0 && (stamp.Value <= progress[index-1].Value || stamp.Microseconds < progress[index-1].Microseconds) {
					t.Errorf("%s %s keeps %+v as sampler %s's event %d; want its first step, then more of its %d steps each time, in order", run, section, stamp, id, index+1, steps)
				}
			}
			if len(progress) == 0 || progress[len(progress)-1].Value != steps {
				t.Errorf("%s %s keeps the steps %+v of sampler %s; want them to end at its %d", run, section, progress, id, steps)
			}
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
	// agree holds two recordings of one section and its inputs to one request, one install and the same output bytes.
	agree := func(section string, first, second recorded) {
		t.Helper()
		if err := distinctRuns(first, second); err != nil {
			t.Errorf("%s: %v", section, err)
		}
		if !maps.EqualFunc(first.Sigmas, second.Sigmas, slices.Equal) {
			t.Errorf("%s: the two recordings sampled by %v and %v; want the same sigmas", section, first.Sigmas, second.Sigmas)
		}
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
	for _, section := range ultra.titles() {
		agree(section, read("pass-1", section), read("pass-2", section))
	}
	// Text to image and its Turbo section on a real prompt, and prompt mode 3 with the builder filled, each recorded
	// twice (moss-real-prompt).
	for _, section := range []string{"TEXT TO IMAGE", "TEXT TO IMAGE TURBO"} {
		agree(section+" on the real prompt", read("real-1", section), read("real-2", section))
	}
	agree("prompt mode 3 with the builder filled", read("builder-1", "TEXT TO IMAGE"), read("builder-2", "TEXT TO IMAGE"))
	speed := read("own-size", "TEXT TO IMAGE")
	if len(speed.Set) != 0 {
		t.Errorf("the speed recording set %v; want the workflow's own size, nothing set", speed.Set)
	}
	// The workflow's prompt mode is its PROMPT MODE node's value, 1 in the sections as saved.
	for _, mode := range []float64{2, 3} {
		taken := read(fmt.Sprintf("mode-%d", int(mode)), "TEXT TO IMAGE")
		if taken.Set[promptMode] != mode || taken.Set[ultraSizeInput(taken)] != 0.5625 || len(taken.Texts) == 0 {
			t.Errorf("the prompt mode %d recording set %v and kept the texts %v; want %s at %d, 1024 x 576, and its prompts' texts", int(mode), taken.Set, taken.Texts, promptMode, int(mode))
		}
	}
	if slices.Sort(readFiles); !slices.Equal(readFiles, recordingFiles) {
		t.Errorf("the test read %v; want the recordings it names, %v", readFiles, recordingFiles)
	}
}

// promptMode is the input that selects the workflow's prompt mode: its PROMPT MODE node's value.
const promptMode = "483.value"

// recordingFiles are the recordings the test reads, each named whole, as every fixture is by its reader.
var recordingFiles = []string{
	"fixtures/comfy/moss-ultra/recordings/builder-1/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/builder-2/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/mode-2/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/mode-3/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/own-size/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/pass-1/depth-control-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/pass-1/depth-control.json",
	"fixtures/comfy/moss-ultra/recordings/pass-1/image-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/pass-1/image-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/pass-1/text-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/pass-1/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/pass-2/depth-control-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/pass-2/depth-control.json",
	"fixtures/comfy/moss-ultra/recordings/pass-2/image-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/pass-2/image-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/pass-2/text-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/pass-2/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/real-1/text-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/real-1/text-to-image.json",
	"fixtures/comfy/moss-ultra/recordings/real-2/text-to-image-turbo.json",
	"fixtures/comfy/moss-ultra/recordings/real-2/text-to-image.json",
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
