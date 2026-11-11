//go:build windows && modeltest

package main

import (
	"bytes"
	"fmt"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The pictures a judge beside a resident video model is timed on: residencyFrames frames of residencyClips VidGen
// captions the Wan student draws at its suite's size, each asked whether it is sharp, 12 questions in all.
const (
	residencyClips  = 2
	residencySeed   = 29
	residencyCommit = "judge-residency"
	// residencyRootEnv names, in the processes the residency test starts, the directory the pictures, their suite
	// and the time alone are written in, and residencyStageEnv which stage the process runs.
	residencyRootEnv  = "OVERGO_RESIDENCY_ROOT"
	residencyStageEnv = "OVERGO_RESIDENCY_STAGE"
	residencyAlone    = "alone.json"
	// residencyBound is how many times its time alone a question may take beside the resident video model.
	residencyBound = 1.5
)

var residencyFrames = []int{10, 20, 30, 40, 50, 60}

// judgedTime is how long a judge took a question, and where its weights were placed when it opened.
type judgedTime struct {
	Seconds   float64                  `json:"seconds"`
	Residency recipe.RealizedResidency `json:"residency"`
}

// drawResidencyPictures has the Wan student draw residencyClips VidGen captions in this process into directory, and
// writes residencyFrames frames of each under root as pictures, with the suite that asks the judge of each whether it
// is sharp. The student's sessions stay resident in the process that draws.
func drawResidencyPictures(t *testing.T, roots dataroot.Roots, root, directory string) evaluation.GeneratedAnswerSuite {
	t.Helper()
	ctx := t.Context()
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	items := testcheck.Must(evaluation.PickVideoCaptions(filepath.Join(roots.Datasets, "VidGen", "VidGen_1M_video_caption.json"), residencyClips, residencySeed))(t)
	prompts := make([]renderPrompt, len(items))
	for index, item := range items {
		prompts[index] = renderPrompt{ID: item.ID, Prompt: item.Caption}
	}
	settings := map[string]float64{"width": judgedWidth, "height": judgedHeight, "frames": motionFrames}
	keys := testcheck.Must(renderMedia(ctx, roots.Store, roots.ResolveModelPath("Self-Forcing"), recipe.TaskVideoGen, directory, prompts, settings, "", residencyCommit))(t)
	var manifest renderManifest
	testcheck.NoError(t, jsonfile.Decode(filepath.Join(directory, renderManifestFile), &manifest))
	suite := evaluation.GeneratedAnswerSuite{Kind: "generated-answer", Schema: "overgo/judge-residency/v1", Source: "judge-speed/residency",
		Transforms: []string{"trim-space", "lowercase"}}
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "frames"), renderDirectoryMode))
	for _, item := range items {
		frames := testcheck.Must(media.DecodeVideoFile(ctx, filepath.Join(directory, manifest.Draws[keys[item.ID]].File), ffmpeg, motionFrameRate, motionFrames))(t)
		for _, at := range residencyFrames {
			var encoded bytes.Buffer
			testcheck.NoError(t, png.Encode(&encoded, frames[at]))
			name := fmt.Sprintf("frames/%s-%d.png", item.ID, at)
			testcheck.NoError(t, os.WriteFile(filepath.Join(root, name), encoded.Bytes(), renderFileMode))
			suite.Cases = append(suite.Cases, evaluation.GeneratedAnswerCase{
				Name:      fmt.Sprintf("%s/%d", item.ID, at),
				Prompt:    "<overgo-media kind=image name=" + name + ">\n" + fidelityQuestions[0] + "\nA. yes\nB. no\nAnswer with the letter only.",
				MaxTokens: 1, Answers: []string{"A"}, Choices: []string{"A", "B"}, Group: item.ID,
			})
		}
	}
	return suite
}

// timeJudging opens Qwen3.8-27B over root and has it judge suite, and says how long a question took and where its
// weights were placed.
func timeJudging(t *testing.T, roots dataroot.Roots, root string, suite evaluation.GeneratedAnswerSuite) judgedTime {
	t.Helper()
	judge := openCalibration(t, t.Context(), roots, root)
	started := time.Now()
	judge.judged(t, t.Context(), suite)
	return judgedTime{Seconds: time.Since(started).Seconds() / float64(len(suite.Cases)), Residency: judge.session.runner.RealizedResidency()}
}

// TestJudgeResidencyStage is a stage the residency test runs in a process of its own: "draw" draws the pictures and
// writes their suite, and "alone" judges them in a process that holds no video model, writing how long a question
// took. It runs only there.
func TestJudgeResidencyStage(t *testing.T) {
	root, stage := os.Getenv(residencyRootEnv), os.Getenv(residencyStageEnv)
	if root == "" {
		testskip.NotApplicable(t, "the stages run in the processes TestAJudgeBesideAResidentVideoModelAnswersAtItsOwnSpeed starts")
	}
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	switch stage {
	case "draw":
		suite := drawResidencyPictures(t, roots, root, filepath.Join(root, "drawn"))
		testcheck.NoError(t, jsonfile.Write(filepath.Join(root, judgedSuiteFile), suite, renderFileMode))
	case "alone":
		var suite evaluation.GeneratedAnswerSuite
		testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, judgedSuiteFile), &suite))
		testcheck.NoError(t, jsonfile.Write(filepath.Join(root, residencyAlone), timeJudging(t, roots, root, suite), renderFileMode))
	default:
		t.Fatalf("no residency stage %q", stage)
	}
}

// TestAJudgeBesideAResidentVideoModelAnswersAtItsOwnSpeed times Qwen3.8-27B judging the same 12 pictures twice: in a
// process that holds no video model, and in this one after the Wan student has drawn clips here, whose sessions stay
// resident for the process's life, as a server holding both does. On 2026-11-09 the judge took about 20 s a picture
// beside the student and about 2 s alone. A question beside it takes at most residencyBound times its time alone, and
// where each judge's weights were placed is said.
func TestAJudgeBesideAResidentVideoModelAnswersAtItsOwnSpeed(t *testing.T) {
	cudatest.Require(t)
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	root := t.TempDir()
	for _, stage := range []string{"draw", "alone"} {
		var logged bytes.Buffer
		child := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestJudgeResidencyStage$", "-test.v", "-test.timeout=0")
		child.Env = append(os.Environ(), residencyRootEnv+"="+root, residencyStageEnv+"="+stage)
		child.Stdout, child.Stderr = &logged, &logged
		err := child.Run()
		for line := range strings.Lines(logged.String()) {
			t.Log(strings.TrimRight(line, "\r\n"))
		}
		testcheck.NoError(t, err)
	}
	var suite evaluation.GeneratedAnswerSuite
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, judgedSuiteFile), &suite))
	var alone judgedTime
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(root, residencyAlone), &alone))
	started := time.Now()
	drawResidencyPictures(t, roots, t.TempDir(), filepath.Join(t.TempDir(), "drawn"))
	t.Logf("the student drew its clips in this process in %s", time.Since(started).Round(time.Second))
	beside := timeJudging(t, roots, root, suite)
	t.Logf("alone the judge took %.2f s a question, its weights %s; beside the student %.2f s, its weights %s", alone.Seconds, alone.Residency, beside.Seconds, beside.Residency)
	if beside.Seconds > residencyBound*alone.Seconds {
		t.Errorf("beside the resident student the judge took %.2f s a question, %.1f times its %.2f s alone, want at most %.1f times", beside.Seconds, beside.Seconds/alone.Seconds, alone.Seconds, residencyBound)
	}
}
