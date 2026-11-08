package latentvideo

import (
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/dataroot"
	"overgo/internal/tensor"
)

// wanStudentCheckpoint is where a Self-Forcing model directory holds its few-step student, as its publisher lays it
// out (huggingface.co/gdhe17/Self-Forcing).
const wanStudentCheckpoint = "checkpoints/self_forcing_dmd.pt"

// unguidedScale is the guidance of a student: it was distilled from the guided teacher and runs one conditional pass a
// step, reading no negative prompt.
const unguidedScale = 1

// IsWanStudent recognizes a directory holding a few-step student of Wan without loading weights.
func IsWanStudent(path string) bool {
	info, err := os.Stat(filepath.Join(path, filepath.FromSlash(wanStudentCheckpoint)))
	return err == nil && !info.IsDir()
}

// wanStudent is what a student directory runs with: the Wan base it was distilled from, whose text encoder,
// autoencoder and configuration it reads, its own checkpoint, and the chunk-causal policy the catalog declares for it.
type wanStudent struct {
	base, checkpoint string
	policy           editPolicy
}

// resolveWanStudent finds a student directory's base beside it, as LiveEdit's is, or in another of the data root's
// model directories: Self-Forcing sits in the models beside the store and its base in the configured ones.
func resolveWanStudent(modelDirectory string) (wanStudent, error) {
	student := wanStudent{
		base:       filepath.Join(filepath.Dir(modelDirectory), wanBaseDirectory),
		checkpoint: filepath.Join(modelDirectory, filepath.FromSlash(wanStudentCheckpoint)),
	}
	if !IsWan(student.base) {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return wanStudent{}, err
		}
		if student.base = roots.ResolveModelPath(wanBaseDirectory); !IsWan(student.base) {
			return wanStudent{}, fmt.Errorf("latent video: the student in %s finds no %s beside it or in the data root's models", modelDirectory, wanBaseDirectory)
		}
	}
	policy, err := checkpointEditPolicy(student.checkpoint)
	student.policy = policy
	return student, err
}

// WanStudentArtifacts locates the two artifacts a student directory runs with: the Wan base it was distilled from and
// the student's checkpoint inside it.
func WanStudentArtifacts(modelDirectory string) (wanDirectory, checkpoint string, err error) {
	student, err := resolveWanStudent(modelDirectory)
	return student.base, student.checkpoint, err
}

// studentProfile is the base's profile with the student's generation: the clip its policy was trained at, its own
// steps and shift, unguided.
func studentProfile(profile Profile, policy editPolicy) (Profile, error) {
	stride := profile.Policy.VAEStride[tensor.FirstOffset]
	profile.Generation = generationPolicy{
		Frames: (policy.LocalAttentionFrames-1)*stride + 1, Width: policy.SourceWidth, Height: policy.SourceHeight,
		Steps: len(policy.DenoisingTimesteps), Shift: float64(policy.Shift), GuideScale: unguidedScale, Seed: profile.Generation.Seed,
	}
	return videoProfileCodec.New(profile)
}
