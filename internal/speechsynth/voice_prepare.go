package speechsynth

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
	"overgo/internal/tensor"
)

var preparedVoiceContract = artifact.JSONContract(artifact.KindFile, "overgo/prepared-speech-voice/v1")

// voiceScalarBytes accounts for IEEE-754 binary32 preparation buffers.
// Reconsider the width if encoder storage changes from float32.
const voiceScalarBytes = 4

const (
	// SciPy resample_poly defines this Kaiser FIR geometry and computes its taps in binary64.
	// Reconsider these values when the admitted reference processor changes.
	voiceFilterHalfWidths  = 10
	voiceFilterBeta        = 5
	voiceFilterScalarBytes = 8
	// The codec and flow transformers each own qkv plus four width-sized scratch vectors.
	// Reconsider this count if either transformer's workspace changes.
	voiceTransformerScratchVectors = 7
	// A binary32 JSON number and delimiter fit in four binary32 widths.
	// Three text copies cover marshal growth and stored content; reconsider with serialization changes.
	voiceJSONNumberWidths = 4
	voiceJSONCopies       = 3
)

type voiceInput struct {
	Code   string      `json:"implementation"`
	Recipe artifact.ID `json:"recipe"`
	Model  artifact.ID `json:"model"`
	Source artifact.ID `json:"source"`
	Start  int         `json:"start_sample"`
	End    int         `json:"end_sample"`
}

type preparedVoice struct {
	Input      voiceInput  `json:"input"`
	SampleRate int         `json:"sample_rate"`
	Padding    int         `json:"padding"`
	State      *VoiceState `json:"state"`
}

func (r SynthesisRequest) validateVoice() error {
	selected := 0
	for _, present := range []bool{strings.TrimSpace(r.Voice) != "", r.PreparedVoice.Valid(), r.ReferenceAudio.Valid()} {
		if present {
			selected++
		}
	}
	if selected != 1 {
		return errors.New("speechsynth: select exactly one preset voice, prepared voice or reference recording")
	}
	if r.PreparedVoice.Valid() && r.PreparedVoice.Kind() != artifact.KindFile || r.ReferenceAudio.Valid() && r.ReferenceAudio.Kind() != artifact.KindFile {
		return errors.New("speechsynth: voice inputs must identify file artifacts")
	}
	if r.ReferenceAudio.Valid() {
		if r.ReferenceStartSample < 0 || r.ReferenceEndSample <= r.ReferenceStartSample || r.VoiceMemoryBytes <= 0 {
			return errors.New("speechsynth: reference recording requires an explicit nonempty sample range and positive preparation memory budget")
		}
	} else if r.ReferenceStartSample != 0 || r.ReferenceEndSample != 0 || r.VoiceMemoryBytes != 0 {
		return errors.New("speechsynth: sample range and preparation budget require a reference recording")
	}
	return nil
}

func (r SynthesisRequest) voiceKey() string {
	if r.PreparedVoice.Valid() {
		return r.PreparedVoice.String()
	}
	if r.ReferenceAudio.Valid() {
		return fmt.Sprintf("%s/%d/%d", r.ReferenceAudio, r.ReferenceStartSample, r.ReferenceEndSample)
	}
	return strings.TrimSpace(r.Voice)
}

// BindVoice prepares or restores one immutable voice for the selected serving recipe.
// The recipe binds encoder, projection, tokenizer and trained transformer identities.
// Reference ranges use source sample indices and reject stereo rather than silently mixing it.
// The memory budget covers source, tensor and serialization buffers beyond the loaded synthesizer.
// Allocator overhead and model metadata are outside this buffer bound.
// RestoreRecipe must bind the serving surface before a recorded voice is used.
func (s *Synthesizer) BindVoice(ctx context.Context, store artifact.Repository, definition recipe.Definition, request SynthesisRequest) (artifact.ID, error) {
	if err := request.validateVoice(); err != nil {
		return artifact.ID{}, err
	}
	if strings.TrimSpace(request.Voice) != "" {
		return artifact.ID{}, nil
	}
	if ctx == nil || store == nil || !s.complete() {
		return artifact.ID{}, errors.New("speechsynth: voice preparation requires context, store and loaded model")
	}
	if err := context.Cause(ctx); err != nil {
		return artifact.ID{}, err
	}
	if err := definition.ValidateIdentity(); err != nil {
		return artifact.ID{}, err
	}
	if definition.Task != recipe.TaskSpeech || s.boundRecipe != definition.ID {
		return artifact.ID{}, errors.New("speechsynth: voice recipe differs from loaded speech surface")
	}
	inventory, err := Inventory(s.directory)
	if err != nil || inventory.Manifest.ID != definition.Model {
		return artifact.ID{}, errors.Join(errors.New("speechsynth: prepared voice model inventory differs"), err)
	}
	input := voiceInput{Code: capabilityruntime.AnswerScope[*Synthesizer](), Recipe: definition.ID, Model: definition.Model, Source: request.ReferenceAudio, Start: request.ReferenceStartSample, End: request.ReferenceEndSample}
	alias := ""
	id := request.PreparedVoice
	if !id.Valid() {
		key, err := artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/voice-preparation-input/v1"), input)
		if err != nil {
			return artifact.ID{}, err
		}
		alias = "voice.prepared." + key.Descriptor.ID.DigestHex()
		id, _, err = artifact.ResolveAlias(ctx, store, alias)
		if err != nil {
			return artifact.ID{}, err
		}
	}
	var prepared preparedVoice
	if id.Valid() {
		content, err := artifact.RequireTypedContent(ctx, store, id)
		if err != nil {
			return artifact.ID{}, err
		}
		if err := preparedVoiceContract.ValidateContent(content, id); err != nil {
			return artifact.ID{}, err
		}
		if err := strictjson.DecodeBytes(content.Data, &prepared); err != nil {
			return artifact.ID{}, err
		}
		if prepared.Input.Code != input.Code || prepared.Input.Recipe != definition.ID || prepared.Input.Model != definition.Model || request.ReferenceAudio.Valid() && prepared.Input != input {
			return artifact.ID{}, errors.New("speechsynth: prepared voice input or recipe differs")
		}
	} else {
		prepared, err = s.prepareVoice(ctx, store, input, request.VoiceMemoryBytes)
		if err != nil {
			return artifact.ID{}, err
		}
		content, err := artifact.JSONContent(preparedVoiceContract, prepared)
		if err != nil {
			return artifact.ID{}, err
		}
		id = content.Descriptor.ID
		var lineage []artifact.Lineage
		for _, parent := range []artifact.ID{input.Recipe, input.Model, input.Source} {
			lineage = append(lineage, artifact.Lineage{Child: id, Parent: parent, Relation: artifact.RelationDependsOn})
		}
		if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: alias, Contents: []artifact.Content{content}, Lineage: lineage, Aliases: []artifact.AliasBinding{artifact.AliasMove(alias, id, artifact.ID{})}}); err != nil {
			return artifact.ID{}, err
		}
	}
	if prepared.SampleRate != s.model.Codec.SampleRate || prepared.Padding < 0 || prepared.Input.Source.Kind() != artifact.KindFile || prepared.Input.Start < 0 || prepared.Input.End <= prepared.Input.Start {
		return artifact.ID{}, errors.New("speechsynth: prepared voice has invalid recording coordinates")
	}
	if err := prepared.State.validate(s.model.Dims); err != nil {
		return artifact.ID{}, err
	}
	if err := context.Cause(ctx); err != nil {
		return artifact.ID{}, err
	}
	s.voiceMu.Lock()
	if s.voices == nil {
		s.voices = make(map[string]*VoiceState)
	}
	s.voices[request.voiceKey()] = prepared.State
	s.voices[id.String()] = prepared.State
	s.voiceMu.Unlock()
	return id, nil
}

func (s *Synthesizer) prepareVoice(ctx context.Context, store artifact.Repository, input voiceInput, budget int64) (preparedVoice, error) {
	var result preparedVoice
	descriptor, found, err := store.Artifact(ctx, input.Source)
	if err != nil || !found {
		return result, errors.Join(errors.New("speechsynth: reference recording is absent"), err)
	}
	if descriptor.Size >= uint64(budget) {
		return result, errors.New("speechsynth: reference bytes exceed preparation budget")
	}
	content, found, err := artifact.ReadContent(ctx, store, input.Source)
	if err != nil || !found {
		return result, errors.Join(errors.New("speechsynth: reference recording is absent"), err)
	}
	budget -= int64(len(content.Data))
	if budget <= 0 {
		return result, errors.New("speechsynth: reference bytes exceed preparation budget")
	}
	audio, _, err := media.DecodeAudio(ctx, content.Data, uint64(budget/voiceScalarBytes))
	if err != nil {
		return result, err
	}
	if audio.Format.Channels != 1 || input.End > len(audio.Samples) {
		return result, errors.New("speechsynth: reference requires mono audio and a range inside the recording")
	}
	budget -= int64(len(audio.Samples)) * voiceScalarBytes
	config, err := loadConfig(filepath.Join(s.directory, configFileName))
	if err != nil {
		return result, err
	}
	mono, spent, err := resampleVoice(ctx, audio.Samples[input.Start:input.End], int(audio.Format.SampleRate), config.Mimi.SampleRate, budget)
	if err != nil {
		return result, err
	}
	budget -= spent
	source, err := safetensors.OpenSource(s.directory)
	if err != nil {
		return result, err
	}
	defer source.Close()
	shapes, err := source.IntShapes()
	if err != nil {
		return result, err
	}
	read := speechTensorReader(source, shapes)
	boundedRead := func(name string, shape ...int) ([]float32, error) {
		count, ok := checked.ProductInt(shape...)
		if !ok || int64(count) > budget/(tensor.PairedExtent*voiceScalarBytes) {
			return nil, errors.New("speechsynth: encoder weights exceed preparation budget")
		}
		budget -= int64(count) * tensor.PairedExtent * voiceScalarBytes
		return read(name, shape...)
	}
	encoder, err := loadCodecEncoder(boundedRead, shapes, config, s.model.Dims.DModel, s.model.Normalization)
	if err != nil {
		return result, err
	}
	if err := encoder.admit(len(mono), s.model.Dims, budget); err != nil {
		return result, err
	}
	conditioning, padding, err := encoder.encode(ctx, mono)
	if err != nil {
		return result, err
	}
	frames := len(conditioning) / s.model.Dims.DModel
	state := s.model.NewDecodeState(frames)
	for frame := range frames {
		if err := context.Cause(ctx); err != nil {
			return result, err
		}
		s.model.AppendForward(state, conditioning[frame*s.model.Dims.DModel:(frame+1)*s.model.Dims.DModel], 1)
	}
	voice := &VoiceState{Frames: frames, K: state.k, V: state.v}
	if err := voice.validate(s.model.Dims); err != nil {
		return result, err
	}
	return preparedVoice{Input: input, SampleRate: config.Mimi.SampleRate, Padding: padding, State: voice}, nil
}

// resampleVoice matches PocketTTS's scipy.signal.resample_poly default FIR.
// SciPy specifies a Kaiser beta of 5 and a half-width of 10 times the reduced rate.
// Reconsider those source constants when the admitted reference processor changes.
func resampleVoice(ctx context.Context, samples []float32, from, to int, budget int64) ([]float32, int64, error) {
	if from <= 0 || to <= 0 {
		return nil, 0, errors.New("speechsynth: reference sample rates must be positive")
	}
	if ctx == nil {
		return nil, 0, errors.New("speechsynth: resampling requires context")
	}
	if err := context.Cause(ctx); err != nil {
		return nil, 0, err
	}
	a, b := from, to
	for b != 0 {
		a, b = b, a%b
	}
	down, up := from/a, to/a
	if up == down {
		return samples, 0, nil
	}
	product, valid := checked.MulInt(len(samples), up)
	half, validHalf := checked.MulInt(voiceFilterHalfWidths, max(up, down))
	if !valid || product > math.MaxInt-(down-1) || !validHalf || half > (math.MaxInt-1)/tensor.PairedExtent {
		return nil, 0, errors.New("speechsynth: resampling extent overflows")
	}
	count := (product + down - 1) / down
	tapBytes, valid := checked.MulInt(tensor.PairedExtent*half+1, voiceFilterScalarBytes)
	if !valid || int64(tapBytes) > budget || int64(count) > (budget-int64(tapBytes))/voiceScalarBytes {
		return nil, 0, errors.New("speechsynth: resampling exceeds preparation budget")
	}
	spent := int64(count)*voiceScalarBytes + int64(tapBytes)
	bessel := func(x float64) float64 {
		sum, term := float64(1), float64(1)
		for n := 1; ; n++ {
			term *= x * x / (tensor.PairedExtent * tensor.PairedExtent * float64(n) * float64(n))
			next := sum + term
			if next == sum {
				return sum
			}
			sum = next
		}
	}
	taps := make([]float64, tensor.PairedExtent*half+1)
	cutoff, sum := 1/float64(max(up, down)), float64(0)
	for i := range taps {
		x := float64(i-half) * cutoff
		sinc := float64(1)
		if x != 0 {
			sinc = math.Sin(math.Pi*x) / (math.Pi * x)
		}
		u := float64(i-half) / float64(half)
		taps[i] = cutoff * sinc * bessel(voiceFilterBeta*math.Sqrt(max(0, 1-u*u))) / bessel(voiceFilterBeta)
		sum += taps[i]
	}
	for i := range taps {
		taps[i] = float64(float32(taps[i]/sum) * float32(up))
	}
	values, err := media.ResamplePoly(ctx, nil, samples, up, down, taps)
	return values, spent, err
}

// admit bounds preparation buffers from encoder geometry before allocation.
// The bound sums transient buffers rather than assuming garbage collection releases them promptly.
func (e *codecEncoder) admit(samples int, dims Dims, budget int64) error {
	remaining := budget / voiceScalarBytes
	reserve := func(factors ...int) bool {
		count, ok := checked.ProductInt(factors...)
		if !ok || int64(count) > remaining {
			return false
		}
		remaining -= int64(count)
		return true
	}
	if samples <= 0 || samples > math.MaxInt-e.frameSamples {
		return errors.New("speechsynth: invalid reference extent")
	}
	frames := (samples + e.frameSamples - 1) / e.frameSamples * e.frameSamples
	ok := reserve(frames)
	for index, cv := range e.convs {
		if index > 0 && index <= len(e.resnets) {
			ok = reserve(frames, cv.cIn) && ok
			for _, residual := range e.resnets[index-1] {
				ok = reserve(frames, residual.cOut) && ok
			}
		}
		frames /= cv.strd
		ok = reserve(frames, cv.cOut) && ok
	}
	c := e.transform
	ok = reserve(tensor.PairedExtent, tensor.PairedExtent, len(c.layers), c.window, c.d) && ok
	// The codec transformer owns qkv, norm, attention, projection, current row and hidden buffers.
	ok = reserve(voiceTransformerScratchVectors, c.d) && reserve(c.ff) && ok
	ok = reserve(frames, len(c.layers), c.h, c.window) && ok
	ok = reserve(e.down.cIn, frames+e.down.k-e.down.strd) && ok
	frames /= e.down.strd
	ok = reserve(tensor.PairedExtent, frames, e.down.cOut) && ok
	frames += len(e.bos) / e.width
	ok = reserve(frames, e.width) && ok
	// Each row's flow prefill owns qkv, query, norm, attention, projection and hidden buffers.
	ok = reserve(frames, voiceTransformerScratchVectors, dims.DModel) && reserve(frames, dims.FF) && ok
	ok = reserve(frames, frames, dims.Heads, dims.Layers) && ok
	state, valid := checked.ProductInt(tensor.PairedExtent, frames, dims.Layers, dims.DModel)
	ok = valid && reserve(state, 1+voiceJSONNumberWidths*voiceJSONCopies) && ok
	if !ok || int64(state)*voiceJSONNumberWidths*voiceScalarBytes >= artifact.MaxContentBytes {
		return errors.New("speechsynth: voice workspace or prepared artifact exceeds its declared limit")
	}
	return nil
}
