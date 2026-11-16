package media

import (
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// standInFFmpeg is a program that lists, as ffmpeg -encoders does, the encoders the file beside it names.
const standInFFmpeg = `package main

import (
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	self, _ := os.Executable()
	listed, _ := os.ReadFile(filepath.Join(filepath.Dir(self), "encoders"))
	fmt.Print("Encoders:\n ------\n" + string(listed))
}
`

// TestTheClipRuntimesAreFoundWithoutASessionSetting holds the runtimes a clip is encoded and finished with to the
// record the data root keeps (clip-runtimes/do): on 2026-11-15 a person's clip, with OVERGO_FFMPEG unset, was encoded
// on libx264 in 167 s where NVENC takes 8.3 s, and the cycle's lanes refused after about five hours for each of the two
// settings a session left out. With both settings empty, the FFmpeg and the effects runtime the record names are the
// ones found: a clip that asks for hevc_nvenc gets the build that lists it, and one that asks for nothing the first the
// machine holds, the recorded one; among two builds, one asking for hevc_nvenc gets the second, which lists it. A
// runtime that misses a file is refused with the command that records another, and with no record the directory
// NVIDIA's installer names is the one found. No production code reads either setting.
func TestTheClipRuntimesAreFoundWithoutASessionSetting(t *testing.T) {
	// Serial: it names the data root and empties the two settings, both read from the process's environment.
	root := t.TempDir()
	t.Setenv("OVERGO_DATA_ROOT", root)
	t.Setenv("OVERGO_FFMPEG", "")
	t.Setenv("OVERGO_NVVFX", "")
	source := filepath.Join(t.TempDir(), "ffmpeg")
	testutil.WriteTextFile(t, source, "go.mod", "module standin\n\ngo 1.26\n")
	testutil.WriteTextFile(t, source, "main.go", standInFFmpeg)
	built := filepath.Join(source, "ffmpeg.exe")
	build := exec.CommandContext(t.Context(), "go", "build", "-o", built, ".")
	build.Dir = source
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build the stand-in FFmpeg: %v: %s", err, output)
	}
	standIn := func(name, encoders string) string {
		directory := filepath.Join(root, name)
		testutil.WriteTextFile(t, directory, "encoders", encoders)
		path := filepath.Join(directory, "ffmpeg.exe")
		testcheck.NoError(t, os.WriteFile(path, testcheck.Must(os.ReadFile(built))(t), fs.FileMode(0o755)))
		return path
	}
	plain := standIn("plain", " V....D libx264              libx264 H.264\n")
	nvenc := standIn("nvenc", " V....D libx264              libx264 H.264\n V....D hevc_nvenc           NVIDIA NVENC hevc encoder\n")
	files := []string{"NVVideoEffects.dll", "NVCVImage.dll"}
	effects := filepath.Join(root, "effects")
	for _, file := range files {
		testutil.WriteTextFile(t, effects, file, "library")
	}

	record := testcheck.Must(RecordClipRuntimes(ClipRuntimes{FFmpeg: nvenc, Effects: effects}, files))(t)
	if want := filepath.Join(root, clipRuntimesFile); record != want {
		t.Errorf("the record was written to %s; want %s, beside the data root's store", record, want)
	}
	if finished, served := testcheck.Must(ResolveFFmpeg("", "hevc_nvenc"))(t), testcheck.Must(FFmpeg())(t); finished != nvenc || served != nvenc {
		t.Errorf("with no setting a finished clip's FFmpeg is %s and a served clip's %s; want the recorded %s for both", finished, served, nvenc)
	}
	if asked, first := testcheck.Must(chooseFFmpeg([]string{plain, nvenc}, []string{"hevc_nvenc"}))(t),
		testcheck.Must(chooseFFmpeg([]string{plain, nvenc}, nil))(t); asked != nvenc || first != plain {
		t.Errorf("of %s and %s, a clip asking for hevc_nvenc gets %s and one asking for nothing %s; want the second, then the first", plain, nvenc, asked, first)
	}
	if found := testcheck.Must(EffectsRuntime(files))(t); found != effects {
		t.Errorf("with no setting the effects runtime is %s; want the recorded %s", found, effects)
	}
	if _, err := EffectsRuntime(append(files, "nvVFXVideoSuperRes.dll")); err == nil || !strings.Contains(err.Error(), "-clip-runtimes") {
		t.Errorf("a recorded runtime that misses a file answered %v; want a refusal naming the command that records another", err)
	}
	if installed := testcheck.Must(effectsFrom(ClipRuntimes{}, record, effects, files))(t); installed != effects {
		t.Errorf("with no record the effects runtime is %s; want %s, the directory its installer names", installed, effects)
	}

	repository := testutil.RepoRoot(t)
	for _, tree := range []string{"internal", "cmd"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(repository, tree), func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return err
			}
			text := string(testcheck.Must(os.ReadFile(path))(t))
			for _, setting := range []string{"OVERGO_FFMPEG", "OVERGO_NVVFX"} {
				if strings.Contains(text, `"`+setting+`"`) {
					t.Errorf("%s reads %s, a setting a session would have to name", path, setting)
				}
			}
			return nil
		}))
	}
}
