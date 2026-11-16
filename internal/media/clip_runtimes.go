package media

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"overgo/internal/dataroot"
	"overgo/internal/envsetting"
	"overgo/internal/jsonfile"
	"overgo/internal/processcontrol"
)

// ClipRuntimes is the record the data root keeps, beside its store, of the runtimes a clip is encoded and finished with
// where the machine holds them on no search path: an FFmpeg, and the directory of NVIDIA's Video Effects runtime,
// read where it is installed and never copied. It is written once (go run ./cmd/generate -clip-runtimes) and read
// wherever a clip is written, so no session names the runtimes: on 2026-11-15 the cycle's lanes refused after about
// five hours for each of two settings a session left out, and a person's clip, unset, was encoded on libx264 in 167 s
// where NVENC takes 8.3 s (clip-runtimes/do).
type ClipRuntimes struct {
	FFmpeg  string `json:"ffmpeg,omitzero"`
	Effects string `json:"effects,omitzero"`
}

// clipRuntimesFile is the record's name beside the data root's store.
const clipRuntimesFile = "clip-runtimes.json"

// RecordClipRuntimesCommand is the command that writes the record, which a refusal for a runtime not found names.
const RecordClipRuntimesCommand = "go run ./cmd/generate -clip-runtimes -ffmpeg <FFmpeg executable> -effects <directory of NVIDIA's Video Effects runtime>"

// clipRuntimesPath is where the current data root keeps the record.
func clipRuntimesPath() (string, error) {
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(roots.Store), clipRuntimesFile), nil
}

// ReadClipRuntimes reads the record the current data root keeps, empty where it keeps none, and where it is.
func ReadClipRuntimes() (ClipRuntimes, string, error) {
	path, err := clipRuntimesPath()
	if err != nil {
		return ClipRuntimes{}, "", err
	}
	held, err := clipRuntimesAt(path)
	return held, path, err
}

// clipRuntimesAt reads the record at path, empty where there is none.
func clipRuntimesAt(path string) (ClipRuntimes, error) {
	var held ClipRuntimes
	if err := jsonfile.DecodeStrict(path, &held); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return ClipRuntimes{}, fmt.Errorf("media: the clip runtimes' record %s: %w", path, err)
	}
	return held, nil
}

// RecordClipRuntimes writes the record into the current data root, each runtime it names checked first: an FFmpeg
// that runs and lists its encoders, and a directory that holds every file of files. It says where it wrote.
func RecordClipRuntimes(runtimes ClipRuntimes, files []string) (string, error) {
	if runtimes.FFmpeg != "" {
		if _, err := Encoders(runtimes.FFmpeg); err != nil {
			return "", err
		}
	}
	if runtimes.Effects != "" {
		if err := holdsEvery(runtimes.Effects, files); err != nil {
			return "", err
		}
	}
	path, err := clipRuntimesPath()
	if err != nil {
		return "", err
	}
	return path, jsonfile.Write(path, runtimes, fs.FileMode(0o644))
}

// ffmpegCandidates are the FFmpegs the machine holds, in the order one is chosen: the one the record names, one on the
// path, and the one a Windows download helper installs. One the record names that is not there is refused by name,
// since the record says the machine holds it.
func ffmpegCandidates(recorded ClipRuntimes, record string) ([]string, error) {
	var found []string
	if recorded.FFmpeg != "" {
		if _, err := os.Stat(recorded.FFmpeg); err != nil {
			return nil, fmt.Errorf("media: the FFmpeg %s names is not there, record another (%s): %w", record, RecordClipRuntimesCommand, err)
		}
		found = append(found, recorded.FFmpeg)
	}
	if path, err := exec.LookPath("ffmpeg"); err == nil {
		found = append(found, path)
	}
	if runtime.GOOS == "windows" {
		if programFiles := envsetting.Result("ProgramFiles"); programFiles != "" {
			if helper := filepath.Join(programFiles, "DownloadHelper CoApp", "ffmpeg.exe"); fileExists(helper) {
				found = append(found, helper)
			}
		}
	}
	return found, nil
}

// fileExists says a file is at path.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// chooseFFmpeg is the first of the candidates whose build lists every encoder asked for, else the first of them, whose
// lack the encoder that writes the clip says.
func chooseFFmpeg(candidates, encoders []string) (string, error) {
	if len(candidates) == 0 {
		return "", fmt.Errorf("media: FFmpeg is unavailable: none is on the path or recorded (%s)", RecordClipRuntimesCommand)
	}
	for _, candidate := range candidates {
		listed, err := Encoders(candidate)
		if err == nil && !slices.ContainsFunc(encoders, func(encoder string) bool { return !slices.Contains(listed, encoder) }) {
			return candidate, nil
		}
	}
	return candidates[0], nil
}

// Encoders are the encoders an FFmpeg lists (ffmpeg -encoders), read once for each executable.
var Encoders = func() func(string) ([]string, error) {
	var mu sync.Mutex
	read := map[string][]string{}
	return func(ffmpeg string) ([]string, error) {
		mu.Lock()
		defer mu.Unlock()
		if listed, known := read[ffmpeg]; known {
			return listed, nil
		}
		var output, diagnostic bytes.Buffer
		receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{Path: ffmpeg,
			Args: []string{"-hide_banner", "-encoders"}, Stdout: &output, Stderr: &diagnostic})
		if err == nil && receipt.ExitCode != 0 {
			err = fmt.Errorf("media: %s -encoders exited %d: %s", ffmpeg, receipt.ExitCode, diagnostic.String())
		}
		if err != nil {
			return nil, err
		}
		var listed []string
		for line := range strings.Lines(output.String()) {
			// Each encoder's line is its capability flags, then its name.
			if fields := strings.Fields(line); len(fields) > 1 {
				listed = append(listed, fields[1])
			}
		}
		read[ffmpeg] = listed
		return listed, nil
	}
}()

// effectsInstalled is the setting NVIDIA's Video Effects installer writes, naming the directory it installed the
// runtime to.
const effectsInstalled = "NV_VIDEO_EFFECTS_PATH"

// EffectsRuntime is the directory of NVIDIA's Video Effects runtime that holds every file of files: the one the data
// root's record names, else the one its installer names. One that misses a file, or none found, is refused by name,
// with the command that records it.
func EffectsRuntime(files []string) (string, error) {
	recorded, record, err := ReadClipRuntimes()
	if err != nil {
		return "", err
	}
	return effectsFrom(recorded, record, envsetting.Result("NV_VIDEO_EFFECTS_PATH"), files)
}

// effectsFrom is the effects runtime's directory the record names, else the one installed names.
func effectsFrom(recorded ClipRuntimes, record, installed string, files []string) (string, error) {
	switch {
	case recorded.Effects != "":
		if err := holdsEvery(recorded.Effects, files); err != nil {
			return "", fmt.Errorf("media: NVIDIA's Video Effects runtime %s names: %w; record another (%s)", record, err, RecordClipRuntimesCommand)
		}
		return recorded.Effects, nil
	case installed != "":
		if err := holdsEvery(installed, files); err != nil {
			return "", fmt.Errorf("media: NVIDIA's Video Effects runtime %s names: %w", effectsInstalled, err)
		}
		return installed, nil
	}
	return "", fmt.Errorf("media: NVIDIA's Video Effects runtime is neither recorded in %s nor installed (%s): record it once (%s)", record, effectsInstalled, RecordClipRuntimesCommand)
}

// holdsEvery refuses a directory that misses a file of files, by name.
func holdsEvery(directory string, files []string) error {
	for _, file := range files {
		if _, err := os.Stat(filepath.Join(directory, file)); err != nil {
			return fmt.Errorf("%s has no %s: %w", directory, file, err)
		}
	}
	return nil
}
