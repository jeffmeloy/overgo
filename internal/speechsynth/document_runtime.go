package speechsynth

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
)

var documentPlanContract = artifact.JSONContract(artifact.KindFile, "overgo.speech-document.v1")
var documentWAVContract = artifact.DocumentContract{Kind: artifact.KindOutput, MediaType: media.WAVMediaType, Schema: "overgo.speech-document.wav.v1"}

// DocumentWAV is a completed recording stored directly in its download format.
type DocumentWAV struct{ Data []byte }

// DecodeContent restores the stored WAV for an identical completed operation.
func (wave *DocumentWAV) DecodeContent(content artifact.Content) error {
	if content.Descriptor.MediaType != media.WAVMediaType {
		return errors.New("speech document: recorded output is not WAV")
	}
	wave.Data = slices.Clone(content.Data)
	return nil
}

// Content publishes the completed recording in its download format.
func (wave DocumentWAV) Content() (artifact.Content, error) {
	return documentWAVContract.OwnedContentBytes(wave.Data)
}

// DocumentProgram composes the admitted parent's native stages. Its result is
// still a candidate: callers must use existing recipe admission before running
// or exposing it as a capability.
func DocumentProgram(parent recipe.Program, plan DocumentPlan) (recipe.Program, artifact.Content, map[recipe.PortName]workflowruntime.Value, error) {
	var zero recipe.Program
	base, stages := parent.Definition(), parent.Stages()
	if base.Task != recipe.TaskSpeech || len(stages) != 3 || len(base.Inputs) != 1 || len(base.Outputs) != 1 || len(plan.Segments) == 0 {
		return zero, artifact.Content{}, nil, errors.New("speech document: unsupported parent graph or empty plan")
	}
	for i, module := range []recipe.ModuleID{modelrecipe.ModuleSpeechTokenize, modelrecipe.ModuleSpeechGenerate, modelrecipe.ModuleSpeechDecode} {
		if stages[i].Module.ID != module || stages[i].Node.ID != []recipe.NodeID{"tokenize", "generate", "decode"}[i] {
			return zero, artifact.Content{}, nil, errors.New("speech document: parent stages differ from native speech")
		}
	}
	content, err := artifact.JSONContent(documentPlanContract, plan)
	if err != nil {
		return zero, artifact.Content{}, nil, err
	}
	var nodes []recipe.Node
	var edges []recipe.Edge
	inputs := []recipe.Input{{Name: "document", Data: recipe.DataText, Target: recipe.Endpoint{Node: "assemble", Port: "document"}}}
	values := map[recipe.PortName]workflowruntime.Value{"document": workflowruntime.ArtifactValue(recipe.DataText, plan, content)}
	prior := 0
	for i, part := range plan.Segments {
		if part.Start != prior || part.End <= part.Start || part.End > len(plan.Request.Text) || part.MaxFrames <= 0 {
			return zero, artifact.Content{}, nil, errors.New("speech document: incomplete source coverage")
		}
		prior = part.End
		prefix := documentPrefix(i)
		for _, stage := range stages {
			node := stage.Node
			node.ID = recipe.NodeID(prefix + string(node.ID))
			nodes = append(nodes, node)
		}
		for _, edge := range base.Edges {
			edge.From.Node = recipe.NodeID(prefix + string(edge.From.Node))
			edge.To.Node = recipe.NodeID(prefix + string(edge.To.Node))
			edges = append(edges, edge)
		}
		input := base.Inputs[0]
		input.Name = recipe.PortName(prefix + "text")
		input.Target.Node = recipe.NodeID(prefix + string(input.Target.Node))
		inputs = append(inputs, input)
		output := base.Outputs[0].Source
		output.Node = recipe.NodeID(prefix + string(output.Node))
		edges = append(edges, recipe.Edge{From: output, To: recipe.Endpoint{Node: "assemble", Port: "segments"}})
		request := documentSegmentRequest(plan, part)
		segmentContent, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.speech-input.v1"), request)
		if err != nil {
			return zero, artifact.Content{}, nil, err
		}
		values[input.Name] = workflowruntime.ArtifactValue(input.Data, request, segmentContent)
	}
	if prior != len(plan.Request.Text) {
		return zero, artifact.Content{}, nil, errors.New("speech document: trailing source omitted")
	}
	nodes = append(nodes, recipe.Node{ID: "assemble", Module: modelrecipe.ModuleSpeechAssemble, Placement: recipe.PlacementHost})
	dependencies := append(slices.Clone(base.Dependencies), recipe.Dependency{Role: recipe.DependencyExecutionRecipe, Artifact: base.ID})
	definition, err := recipe.NewDefinitionWithDependencies(recipe.TaskSpeech, dependencies, nodes, edges, inputs, []recipe.Output{{Name: "audio", Data: recipe.DataAudio, Source: recipe.Endpoint{Node: "assemble", Port: "audio"}}})
	if err != nil {
		return zero, artifact.Content{}, nil, err
	}
	program, err := modelrecipe.CompileCapability(definition)
	return program, content, values, err
}

func documentPrefix(index int) string { return fmt.Sprintf("part%d-", index) }

func documentSegmentRequest(plan DocumentPlan, part DocumentSegment) SynthesisRequest {
	return SynthesisRequest{Text: plan.Request.Text[part.Start:part.End], Voice: plan.Request.Voice, Seed: plan.Request.Seed, Temperature: plan.Request.Temperature, MaxFrames: part.MaxFrames}
}

// RegisterDocumentRuntime retains native generation and decode receipts and
// adds one ordered assembly stage. Incidental many-input arrival order never
// determines the order of speech in the recording.
func RegisterDocumentRuntime(runtime *workflowruntime.Runtime, store artifact.Repository, modelID artifact.ID, synth *Synthesizer) error {
	if err := RegisterRuntime(runtime, modelID, synth); err != nil {
		return err
	}
	return workflowruntime.RegisterResolvedStage(runtime, modelrecipe.ModuleSpeechAssemble, modelID, func(ctx context.Context, request workflowruntime.StepRequest) (DocumentWAV, error) {
		plan, err := workflowruntime.ScalarInput[DocumentPlan](request, "document")
		if err != nil {
			return DocumentWAV{}, err
		}
		verified, err := synth.PlanDocument(ctx, plan.Request)
		if err != nil {
			return DocumentWAV{}, err
		}
		if !reflect.DeepEqual(verified, plan) {
			return DocumentWAV{}, errors.New("speech document: source plan differs from native policy")
		}
		byArtifact := make(map[artifact.ID]workflowruntime.Datum)
		for _, datum := range request.Inputs["segments"].Items {
			byArtifact[datum.Artifact.ID] = datum
		}
		parts := make([][]float32, 0, len(plan.Segments))
		rate, samples := 0, 0
		// A PCM16 mono WAV has a 44-byte RIFF/fmt/data header and two bytes
		// per sample. Bound the final artifact before the encoder allocates it.
		for i := range plan.Segments {
			if err := context.Cause(ctx); err != nil {
				return DocumentWAV{}, err
			}
			expected, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "overgo.speech-input.v1"), documentSegmentRequest(plan, plan.Segments[i]))
			if err != nil {
				return DocumentWAV{}, err
			}
			inputReceipt, found, err := runrecord.ResolveStageReceipt(ctx, store, request.Operation, recipe.NodeID(documentPrefix(i)+"tokenize"))
			if err != nil {
				return DocumentWAV{}, err
			}
			if !found || len(inputReceipt.Inputs) != 1 || !slices.Equal(inputReceipt.Inputs[0].Artifacts, []artifact.ID{expected.Descriptor.ID}) {
				return DocumentWAV{}, errors.New("speech document: segment request differs from source plan")
			}
			receipt, found, err := runrecord.ResolveStageReceipt(ctx, store, request.Operation, recipe.NodeID(documentPrefix(i)+"decode"))
			if err != nil {
				return DocumentWAV{}, err
			}
			if !found || receipt.State != runrecord.StageCompleted || len(receipt.Outputs) != 1 || len(receipt.Outputs[0].Artifacts) != 1 {
				return DocumentWAV{}, errors.New("speech document: missing ordered decode receipt")
			}
			datum, found := byArtifact[receipt.Outputs[0].Artifacts[0]]
			if !found {
				return DocumentWAV{}, errors.New("speech document: audio differs from ordered receipt")
			}
			part, err := workflowruntime.ScalarInput[Audio](workflowruntime.StepRequest{Inputs: map[recipe.PortName]workflowruntime.Value{"audio": {Kind: recipe.DataAudio, Items: []workflowruntime.Datum{datum}}}}, "audio")
			if err != nil {
				return DocumentWAV{}, err
			}
			if !part.Complete || part.Channels != 1 || part.SampleRate <= 0 || len(part.PCM) == 0 {
				return DocumentWAV{}, fmt.Errorf("speech document: segment %d did not finish", i+1)
			}
			if part.SampleRate != plan.SampleRate || len(part.PCM) > plan.Segments[i].MaxSamples {
				return DocumentWAV{}, errors.New("speech document: decoded audio exceeds the source plan")
			}
			if rate != 0 && rate != part.SampleRate {
				return DocumentWAV{}, errors.New("speech document: segment sample rate changed")
			}
			if len(part.PCM) > (artifact.MaxContentBytes-documentWAVHeaderBytes)/documentSampleBytes-samples {
				return DocumentWAV{}, errors.New("speech document: recording exceeds WAV artifact limit")
			}
			rate = part.SampleRate
			samples += len(part.PCM)
			parts = append(parts, part.PCM)
		}
		data, err := encodeDocumentWAV(parts, rate)
		return DocumentWAV{Data: data}, err
	}, func(wave DocumentWAV) (artifact.Content, error) { return wave.Content() })
}

// Keep PCM quantization in media's encoder; assemble only its canonical WAV
// payloads here without allocating a whole-document float buffer.
func encodeDocumentWAV(parts [][]float32, rate int) ([]byte, error) {
	samples := 0
	for _, part := range parts {
		if len(part) > (artifact.MaxContentBytes-documentWAVHeaderBytes)/documentSampleBytes-samples {
			return nil, errors.New("speech document: recording exceeds WAV artifact limit")
		}
		samples += len(part)
	}
	header, err := media.EncodeWAVPCM16(nil, rate)
	if err != nil {
		return nil, err
	}
	data := make([]byte, documentWAVHeaderBytes+samples*documentSampleBytes)
	copy(data, header)
	offset := documentWAVHeaderBytes
	for _, part := range parts {
		wave, err := media.EncodeWAVPCM16(part, rate)
		if err != nil {
			return nil, err
		}
		offset += copy(data[offset:], wave[documentWAVHeaderBytes:])
	}
	// RIFF size excludes its eight-byte chunk header; data length occupies
	// the final four bytes of the canonical PCM header.
	binary.LittleEndian.PutUint32(data[4:8], uint32(len(data)-8))
	binary.LittleEndian.PutUint32(data[documentWAVHeaderBytes-4:documentWAVHeaderBytes], uint32(len(data)-documentWAVHeaderBytes))
	return data, nil
}
