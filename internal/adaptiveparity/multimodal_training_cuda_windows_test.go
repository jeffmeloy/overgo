//go:build windows

package adaptiveparity_test

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

const (
	matrixTextSHA       = "9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f"
	matrixImageSHA      = "026362ccd74747359902464829a97e1aa1321d745efdee7f2056e226477484c0"
	matrixAudioSHA      = "ac135a0ddeb91a15cd70eec58949af2e329311b0d9848298ae1d9b1284ffc90f"
	matrixAudioLabelSHA = "8749bb8c9e0eaf7e10102315242a35665ad4a583f4eeb3d27f34065013ec140b"
)

func TestMultimodalTrainingMatrix(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	textPath := filepath.Join(roots.Datasets, "wikitext2_train.txt")
	imagePath := filepath.Join(roots.Datasets, "P2datasetFull", "val1", "2", "test2 (934).jpg")
	audioPath := filepath.Join(roots.Datasets, "Marco_Longspeech", "LongSpeech_p1", "wavs", "part_00", "000000.wav")
	audioLabelPath := filepath.Join(roots.Datasets, "Marco_Longspeech", "LongSpeechQA", "speaker_count", "train.jsonl")
	for path, want := range map[string]string{
		textPath: matrixTextSHA, imagePath: matrixImageSHA, audioPath: matrixAudioSHA, audioLabelPath: matrixAudioLabelSHA,
	} {
		if got := qwenFileSHA256(t, path); got != want {
			t.Fatalf("training corpus %s identity=%s, want %s", filepath.Base(path), got, want)
		}
	}

	modelPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-bf16.gguf")
	projectorPath := filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-mmproj-bf16.gguf")
	file, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	vocab, err := tokenizer.Load(file)
	closeErr := file.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("load E4B tokenizer: %v / close: %v", err, closeErr)
	}
	paragraph := realParagraph(t, textPath)
	textInput, textTarget := paragraph[:len(paragraph)/2], paragraph[len(paragraph)/2:]
	textDataset := identifyPath(t, artifact.KindDataset, textPath)
	imageDataset := identifyBytes(t, artifact.KindDataset, matrixImageSHA+"/class/2")
	audioDataset := identifyBytes(t, artifact.KindDataset, matrixAudioSHA+"/"+matrixAudioLabelSHA)
	processedText := processTextPair(t, textDataset, textInput, textTarget)
	pairedInput, pairedTarget, err := trainingdata.TextPair(processedText)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := vocab.Encode(pairedInput+pairedTarget, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil || len(ids) < 4 {
		t.Fatalf("tokenize real text: tokens=%d err=%v", len(ids), err)
	}
	inputIDs, targetIDs, err := trainingdata.AdjacentTokenRows(ids[:min(4, len(ids))])
	if err != nil {
		t.Fatal(err)
	}
	rows := len(inputIDs)

	imageRecord := trainingdata.RawRecord{ID: "p2/934", Group: "p2/heldout", Data: readFile(t, imagePath)}
	imageExample := processMediaTextPair(t, imageDataset, imageRecord,
		trainingdata.ImageProcessor(trainingdata.RoleInput), recipecontract.ModalityImage, "The image class is 2.")
	decodedImage, err := trainingdata.Image(imageExample.Values[0])
	if err != nil {
		t.Fatal(err)
	}
	audioRecord := trainingdata.RawRecord{ID: "longspeech/000000", Group: "longspeech/train", Data: readFile(t, audioPath)}
	audioExample := processMediaTextPair(t, audioDataset, audioRecord,
		trainingdata.AudioProcessor(trainingdata.RoleInput), recipecontract.ModalityAudio,
		realAudioTarget(t, audioLabelPath, "LongSpeech_p1/wavs/part_00/000000.wav"))
	samples, sampleRate, err := trainingdata.Audio(audioExample.Values[0])
	if err != nil {
		t.Fatal(err)
	}
	if maximum := sampleRate * 2; len(samples) > maximum {
		samples = samples[:maximum]
	}

	runner, err := projector.OpenAs[*projector.Gemma4TowerRunner](ctx, projectorPath, projector.OpenOptions{CUDA: true})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	vision, err := runner.EncodeVisionImage(ctx, decodedImage)
	if err != nil {
		t.Fatal(err)
	}
	audioProfile, err := projector.NewAudioProjectionProfile(e4bAudioRopeBase)
	if err != nil {
		t.Fatal(err)
	}
	audio, err := runner.EncodeAudio(ctx, samples, sampleRate, audioProfile)
	if err != nil {
		t.Fatal(err)
	}

	trained, _, err := adaptertrain.LoadArtifact(ctx, modelPath, 0, trainingprogram.BuiltinOptimizerPolicy())
	if err != nil {
		t.Fatal(err)
	}
	examples := make([]adaptertrain.Example, 3)
	examples[0], err = trained.BuildExample(ctx, modelPath, nil, inputIDs, targetIDs, "text-to-text")
	if err != nil {
		t.Fatal(err)
	}
	mediaIDs := func(id uint32, count int) []uint32 {
		result := make([]uint32, count)
		for index := range result {
			result[index] = id
		}
		return result
	}
	hidden := len(examples[0].Input) / rows
	imageTargets := targetTokenIDs(t, vocab, imageExample, 3)
	examples[1], err = trained.BuildExample(ctx, modelPath, vision.Embeddings.Data[:len(imageTargets)*hidden], mediaIDs(258880, len(imageTargets)), imageTargets, "image-to-text")
	if err != nil {
		t.Fatal(err)
	}
	audioTargets := targetTokenIDs(t, vocab, audioExample, 3)
	examples[2], err = trained.BuildExample(ctx, modelPath, audio.Embeddings.Data[:len(audioTargets)*hidden], mediaIDs(258881, len(audioTargets)), audioTargets, "audio-to-text")
	if err != nil {
		t.Fatal(err)
	}

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	measurements := make([]matrixMeasurement, len(examples))
	started := time.Now()
	for index, example := range examples {
		measurements[index].loss, err = trained.Step(worker, example)
		if err != nil || math.IsNaN(measurements[index].loss) || math.IsInf(measurements[index].loss, 0) {
			t.Fatalf("%s loss=%g: %v", example.Kind, measurements[index].loss, err)
		}
		output, err := trained.Output(example)
		if err != nil {
			t.Fatal(err)
		}
		predicted, expected := nearestTargetRows(output, example.Target, example.Rows)
		measurements[index].accuracy, err = trainingprogram.EvaluateNative(trainingprogram.MetricTokenAccuracy, predicted, expected)
		if err != nil || measurements[index].accuracy < 0 || measurements[index].accuracy > 1 {
			t.Fatalf("%s token accuracy=%g: %v", example.Kind, measurements[index].accuracy, err)
		}
	}
	trainWall := time.Since(started)

	store, err := overgodb.Open(filepath.Join(t.TempDir(), "objective-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	objectives := publishMatrixObjectives(t, store, []matrixObjectiveEvidence{
		{name: "e4b-text-to-text", input: recipecontract.ModalityText, dataset: textDataset, corpusSHA: matrixTextSHA, measurement: measurements[0]},
		{name: "e4b-image-to-text", input: recipecontract.ModalityImage, dataset: imageDataset, corpusSHA: matrixImageSHA + "/class/2", projector: projectorPath, measurement: measurements[1]},
		{name: "e4b-audio-to-text", input: recipecontract.ModalityAudio, dataset: audioDataset, corpusSHA: matrixAudioSHA + "/" + matrixAudioLabelSHA, projector: projectorPath, measurement: measurements[2]},
	})
	matrix, err := trainingprogram.CompileObjectiveMatrix(ctx, store, objectives)
	if err != nil {
		t.Fatal(err)
	}
	trainable, refused := 0, 0
	wantTrainable := map[string]bool{"text->text": true, "image->text": true, "audio->text": true}
	for _, row := range matrix {
		key := string(row.Signature.Inputs[0]) + "->" + string(row.Signature.Outputs[0])
		switch row.Disposition {
		case trainingprogram.ObjectiveTrainable:
			if !wantTrainable[key] {
				t.Fatalf("unexpected trainable objective %s", key)
			}
			delete(wantTrainable, key)
			trainable++
		case trainingprogram.ObjectiveRefused:
			if row.Reason == "" || row.Objective.Valid() {
				t.Fatalf("invalid refusal row: %+v", row)
			}
			refused++
		default:
			t.Fatalf("invalid objective disposition: %+v", row)
		}
	}
	if trainable != 3 || refused != 33 {
		t.Fatalf("objective matrix trainable/refused=%d/%d, want 3/33", trainable, refused)
	}
	if len(wantTrainable) != 0 {
		t.Fatalf("trainable objectives absent: %v", wantTrainable)
	}
	t.Logf("real objective matrix: Wikitext=%d bytes P2-image=%d bytes LongSpeech=%d samples; trainable/refused=%d/%d adapter=%d wall=%s metrics=%+v",
		fileSize(t, textPath), fileSize(t, imagePath), len(samples), trainable, refused, trained.ParameterCount(), trainWall, measurements)
}

type matrixObjectiveEvidence struct {
	name, corpusSHA string
	input           recipecontract.Modality
	dataset         artifact.ID
	projector       string
	measurement     matrixMeasurement
}

type matrixMeasurement struct{ loss, accuracy float64 }

func publishMatrixObjectives(t *testing.T, store *overgodb.Store, evidence []matrixObjectiveEvidence) []artifact.ID {
	t.Helper()
	ids := func(kind artifact.Kind, value string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte(value))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	descriptors := map[artifact.ID]artifact.Descriptor{}
	var contents []artifact.Content
	result := make([]artifact.ID, len(evidence))
	for index, item := range evidence {
		dataset := item.dataset
		split := ids(artifact.KindDatasetShard, item.corpusSHA+"/train")
		processor := ids(artifact.KindProfile, "processor/"+string(item.input))
		loss := ids(artifact.KindProfile, "embedding-mse/v1")
		evaluation := ids(artifact.KindProfile, "token-accuracy/v1")
		proof := ids(artifact.KindEvidence, fmt.Sprintf("%s/%s/%.9g/%.9g", item.name, item.corpusSHA, item.measurement.loss, item.measurement.accuracy))
		var projectors []artifact.ID
		if item.projector != "" {
			projectors = []artifact.ID{identifyPath(t, artifact.KindProjector, item.projector)}
		}
		objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
			Name:      item.name,
			Kind:      trainingprogram.ObjectiveTokenPrediction,
			Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{item.input}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
			Dataset:   dataset, Split: split, Processors: []artifact.ID{processor}, Projectors: projectors,
			Loss: loss, Evaluation: evaluation, Metric: trainingprogram.MetricTokenAccuracy,
			Evidence: []artifact.ID{proof}, Authority: trainingprogram.ObjectiveApproved,
		})
		if err != nil {
			t.Fatal(err)
		}
		content, err := objective.Content()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, content)
		result[index] = objective.ID
		for _, id := range append([]artifact.ID{dataset, split, processor, loss, evaluation, proof}, projectors...) {
			descriptors[id] = artifact.Descriptor{ID: id}
		}
	}
	artifacts := make([]artifact.Descriptor, 0, len(descriptors))
	for _, descriptor := range descriptors {
		artifacts = append(artifacts, descriptor)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "real-multimodal-objectives", Artifacts: artifacts, Contents: contents}); err != nil {
		t.Fatal(err)
	}
	return result
}

func processTextPair(t *testing.T, dataset artifact.ID, input, target string) trainingdata.Example {
	t.Helper()
	processor, err := trainingdata.JSONTextPairProcessor("input", "target")
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(map[string]string{"input": input, "target": target})
	if err != nil {
		t.Fatal(err)
	}
	return streamMatrixRecord(t, dataset,
		recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
		[]recipecontract.Modality{recipecontract.ModalityText}, processor,
		trainingdata.RawRecord{ID: "wikitext/0", Group: "wikitext/train", Data: data})
}

func processMediaTextPair(t *testing.T, dataset artifact.ID, record trainingdata.RawRecord, media trainingdata.Processor, modality recipecontract.Modality, target string) trainingdata.Example {
	t.Helper()
	processor := func(ctx context.Context, record trainingdata.RawRecord) (trainingdata.Example, error) {
		example, err := media(ctx, record)
		if err != nil {
			return trainingdata.Example{}, err
		}
		example.Values = append(example.Values, trainingdata.Value{
			Role: trainingdata.RoleTarget, Modality: recipecontract.ModalityText,
			Encoding: trainingdata.EncodingUTF8, Data: []byte(target),
		})
		return example, nil
	}
	return streamMatrixRecord(t, dataset,
		recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{modality}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
		[]recipecontract.Modality{modality, recipecontract.ModalityText}, processor, record)
}

func streamMatrixRecord(t *testing.T, datasetID artifact.ID, signature recipecontract.ModalitySignature, modalities []recipecontract.Modality, processor trainingdata.Processor, record trainingdata.RawRecord) trainingdata.Example {
	t.Helper()
	split, err := artifact.IdentifyBytes(artifact.KindDatasetShard, []byte(datasetID.String()+"/matrix"))
	if err != nil {
		t.Fatal(err)
	}
	profile, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("matrix/"+record.ID))
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := trainingdata.MaterializeRecords(trainingdata.Authority{
		Dataset: datasetID, Split: split, Processors: []artifact.ID{profile}, Signature: signature,
	}, profile, []trainingdata.RawRecord{record}, trainingdata.ProcessorBinding{
		Artifact: profile, Modalities: modalities, Process: processor,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer dataset.Close()
	stream, err := trainingdata.NewStream(dataset, nil)
	if err != nil {
		t.Fatal(err)
	}
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, MicrobatchExamples: 1, DecodeWorkers: 1})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := batcher.Next(t.Context())
	if err != nil || len(batch.Examples) != 1 || batch.State.Position != 1 {
		t.Fatalf("stream matrix record: examples=%d state=%+v err=%v", len(batch.Examples), batch.State, err)
	}
	return batch.Examples[0]
}

func targetTokenIDs(t *testing.T, vocab *tokenizer.Vocab, example trainingdata.Example, maximum int) []uint32 {
	t.Helper()
	var target string
	for _, value := range example.Values {
		if value.Role == trainingdata.RoleTarget && value.Modality == recipecontract.ModalityText && value.Encoding == trainingdata.EncodingUTF8 {
			target = string(value.Data)
		}
	}
	ids, err := vocab.Encode(target, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil || len(ids) == 0 {
		t.Fatalf("tokenize target %q: %d tokens: %v", target, len(ids), err)
	}
	ids = ids[:min(maximum, len(ids))]
	result := make([]uint32, len(ids))
	for index, id := range ids {
		if id < 0 {
			t.Fatal("target produced negative token")
		}
		result[index] = uint32(id)
	}
	return result
}

func realParagraph(t *testing.T, path string) string {
	t.Helper()
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if len(line) >= 256 {
			return line
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	t.Fatal("Wikitext corpus has no usable paragraph")
	return ""
}

func realAudioTarget(t *testing.T, path, audio string) string {
	t.Helper()
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	type message struct {
		Role    string          `json:"role"`
		Audio   string          `json:"audio"`
		Content json.RawMessage `json:"content"`
	}
	var row struct {
		Messages []message `json:"messages"`
	}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil || len(row.Messages) != 2 || row.Messages[0].Audio != audio {
			continue
		}
		var number json.Number
		if err := json.Unmarshal(row.Messages[1].Content, &number); err == nil {
			return number.String()
		}
		var text string
		if err := json.Unmarshal(row.Messages[1].Content, &text); err == nil && text != "" {
			return text
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	t.Fatalf("LongSpeech annotation for %s absent", audio)
	return ""
}

func nearestTargetRows(output, target []float32, rows int) ([]float32, []float32) {
	hidden := len(output) / rows
	predicted, expected := make([]float32, rows), make([]float32, rows)
	for row := range rows {
		best, bestDistance := 0, math.Inf(1)
		for candidate := range rows {
			var distance float64
			for column := range hidden {
				delta := float64(output[row*hidden+column] - target[candidate*hidden+column])
				distance += delta * delta
			}
			if distance < bestDistance {
				best, bestDistance = candidate, distance
			}
		}
		predicted[row], expected[row] = float32(best), float32(row)
	}
	return predicted, expected
}

func identifyPath(t *testing.T, kind artifact.Kind, path string) artifact.ID {
	t.Helper()
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		t.Fatal(err)
	}
	id, _, err := artifact.Identify(kind, file)
	closeErr := file.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("identify %s: %v / close: %v", path, err, closeErr)
	}
	return id
}

func identifyBytes(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	info, err := os.Stat(filepath.Clean(path))
	if err != nil {
		t.Fatal(err)
	}
	return info.Size()
}
