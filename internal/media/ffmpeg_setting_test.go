package media

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestAnUnnamedFFmpegIsTheOneTheSettingNames holds every resolution of FFmpeg to one answer (cycle-lanes/ffmpeg-setting):
// a caller that names none gets the one OVERGO_FFMPEG names, as FFmpeg does, where ResolveFFmpeg("") read the path's
// and the judge read a rendered clip with another FFmpeg than the one that wrote it (cmd/evaluate's
// TestARenderedClipIsAnMP4TheJudgeReads, the cycle's lanes, 2026-11-15). A caller that names one still gets it.
func TestAnUnnamedFFmpegIsTheOneTheSettingNames(t *testing.T) {
	directory := t.TempDir()
	named, other := filepath.Join(directory, "named-ffmpeg.exe"), filepath.Join(directory, "other-ffmpeg.exe")
	for _, path := range []string{named, other} {
		testcheck.NoError(t, os.WriteFile(path, []byte("fixture"), 0o600))
	}
	t.Setenv("OVERGO_FFMPEG", named)
	unnamed, served := testcheck.Must(ResolveFFmpeg(""))(t), testcheck.Must(FFmpeg())(t)
	if unnamed != named || served != named {
		t.Errorf("with OVERGO_FFMPEG naming %s, a caller naming none resolved %s and FFmpeg %s; want the setting's", named, unnamed, served)
	}
	if chosen := testcheck.Must(ResolveFFmpeg(other))(t); chosen != other {
		t.Errorf("a caller naming %s resolved %s; want its own", other, chosen)
	}
}
