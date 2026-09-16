package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/media"
	"overgo/internal/modelcli"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type e4bServingTrace struct {
	*inference.Runner
	ids            []tokenizer.TokenID
	beforeGenerate func(context.Context) error
	afterPrompt    func(context.Context) error
	observationErr error
	onToken        func(inference.TokenEvent) error
	generationErr  error
}

func (r *e4bServingTrace) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	r.ids = slices.Clone(options.PromptTokenIDs)
	if r.ids == nil {
		var err error
		r.ids, err = r.TokenizeText(prompt, true, options.ParseSpecial)
		if err != nil {
			return nil, "", err
		}
	}
	if r.beforeGenerate != nil {
		if err := r.beforeGenerate(ctx); err != nil {
			return nil, "", err
		}
	}
	if r.afterPrompt != nil {
		callback := options.OnPromptEvaluated
		options.OnPromptEvaluated = func(event inference.PromptEvaluation) {
			r.observationErr = r.afterPrompt(ctx)
			if callback != nil {
				callback(event)
			}
		}
	}
	if r.onToken != nil {
		callback := options.OnToken
		options.OnToken = func(event inference.TokenEvent) error {
			if err := r.onToken(event); err != nil {
				return err
			}
			if callback != nil {
				return callback(event)
			}
			return nil
		}
	}
	ids, text, err := r.Runner.Generate(ctx, prompt, options)
	r.generationErr = err
	return ids, text, err
}

// TestE4BHTTPModalities is a serving producer, not dataset or resource acceptance.
// Six input modes traverse all three protocols with one resident model. Expected
// complete outputs come from the independent native SDPA captures, not this server.
func TestE4BHTTPModalities(t *testing.T) {
	cudatest.Require(t)
	started := time.Now()
	publish := os.Getenv("OVERGO_E4B_PUBLISH_PROTOCOL") == "1"
	var revision string
	if publish {
		var err error
		revision, err = runrecord.VerifyingCommit(testutil.RepoRoot(t))
		if err != nil {
			t.Fatal(err)
		}
	}
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var paths []string
	for _, input := range []struct {
		id   string
		kind artifact.LocationKind
	}{
		{"tensor-set:sha256:cd4ada4703c2b76a84a10da94f09b9199b6d4dad7e3dcabbe79d8cee745f4501", artifact.LocationFile},
		{"tensor-set:sha256:185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e", artifact.LocationFile},
		{"dataset:sha256:39b4aa4872600afff4be0633dc066adfe5ae36ac17ade4a6ac4a16e26852b2e3", artifact.LocationDirectory},
	} {
		id, err := artifact.ParseID(input.id)
		if err != nil {
			t.Fatal(err)
		}
		path, err := artifact.AvailablePath(ctx, store, id, input.kind)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	projectionCandidate, err := modelintake.PrepareProjectionCandidate(ctx, store, paths[0], paths[1])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := modelcli.OpenRunner(ctx, roots.Store, paths[0], inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	}()
	if got := runner.ModelID().String(); got != "model:sha256:fb09299dd00edd7ffdcf8cb48e475d2a9c9e30a22c51f79f6d4d793e983c557b" {
		t.Fatalf("wrong model: %s", got)
	}
	if err := runner.RuntimePolicy().ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	projection, err := projector.OpenSession(ctx, paths[1], projector.OpenOptions{CUDA: true, MediaPreprocess: projectionCandidate.Processor})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := projection.Close(); err != nil {
			t.Error(err)
		}
	}()
	if got := projection.Capabilities(); got != (projector.SessionCapabilities{Image: true, MultiImage: true, Audio: true, Video: true, MediaHistory: true}) {
		t.Fatalf("incomplete projection: %+v", got)
	}
	trace := &e4bServingTrace{Runner: runner}
	handler, err := New(Config{RuntimePolicy: runner.RuntimePolicy(), MaxConcurrent: 1, MaxTokens: 128, ImageProjector: projection, AudioProjector: projection, VideoFPS: 1, VideoMaxFrames: 2}, trace)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := handler.Close(); err != nil {
			t.Error(err)
		}
	}()
	cases, oracleBytes := prepareE4BServingCases(t, ctx, paths[2])
	for _, test := range cases {
		for _, route := range []string{"/completion", "/v1/chat/completions", "/v1/responses"} {
			t.Run(test.name+route, func(t *testing.T) {
				encoded := encodeE4BServingRequest(t, runner, test, route)
				response := httptest.NewRecorder()
				started := time.Now()
				handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, route, bytes.NewReader(encoded)).WithContext(ctx))
				if response.Code != http.StatusOK {
					t.Fatalf("HTTP %d: %s", response.Code, response.Body.String())
				}
				var result struct {
					Content string `json:"content"`
					Choices []struct {
						Message struct {
							Content string `json:"content"`
						} `json:"message"`
						FinishReason string `json:"finish_reason"`
					} `json:"choices"`
					Output []struct {
						Content []struct {
							Text string `json:"text"`
						} `json:"content"`
					} `json:"output"`
					Status   string `json:"status"`
					StopType string `json:"stop_type"`
				}
				if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
					t.Fatal(err)
				}
				actual := result.Content
				if route == "/v1/chat/completions" {
					if len(result.Choices) != 1 || result.Choices[0].FinishReason != "stop" {
						t.Fatalf("incomplete output: %s", response.Body.String())
					}
					actual = result.Choices[0].Message.Content
				} else if route == "/v1/responses" {
					if result.Status != "completed" {
						t.Fatalf("incomplete response: %s", response.Body.String())
					}
					for _, item := range result.Output {
						for _, content := range item.Content {
							actual += content.Text
						}
					}
				} else if result.StopType == "limit" {
					t.Fatal("native response exhausted output limit")
				}
				t.Logf("wall=%s complete_output=%q", time.Since(started), actual)
				if !slices.Equal(trace.ids, test.promptIDs) {
					t.Errorf("prepared prompt differs from native reference: got %d tokens, want %d", len(trace.ids), len(test.promptIDs))
				}
				if !slices.Contains(test.outputs, actual) {
					t.Errorf("complete output differs from native reference: got %q, want one of %q", actual, test.outputs)
				}
			})
		}
	}
	t.Log("producer only: exact full outputs over 18 requests; corpus-wide quality, long context, resource soak and canonical projection activation remain separate acceptance obligations")
	if publish && !t.Failed() {
		current, err := runrecord.VerifyingCommit(root)
		if err != nil || current != revision {
			t.Fatalf("protocol producer source changed: %v", err)
		}
		loaded, err := modelrecipe.ResolveActiveGGUF(ctx, store, paths[0])
		if err != nil {
			t.Fatal(err)
		}
		identity, err := loaded.Identity()
		if err != nil {
			t.Fatal(err)
		}
		writer, err := overgodb.Open(roots.Store)
		if err != nil {
			t.Fatal(err)
		}
		defer writer.Close()
		if err := modelintake.RegisterProjectionCandidate(ctx, writer, projectionCandidate); err != nil {
			t.Fatal(err)
		}
		evidence := fmt.Sprintf("contract=e4b-http-modalities;oracle_sha256=%x;model_definition=%s;cases=18", sha256.Sum256(oracleBytes), identity.Definition)
		proof, err := modelintake.PublishVerification(ctx, writer, projectionCandidate.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("published complete HTTP protocol proof: gate=%s run=%s recipe=%s; projection remains unpromoted", proof.Gate, proof.Run, projectionCandidate.Definition.ID)
	}
}

// e4bServingCase binds the exact request preparation to native prompt and output captures.
type e4bServingCase struct {
	name, question string
	media          []string
	promptIDs      []tokenizer.TokenID
	outputs        []string
}

func prepareE4BServingCases(t *testing.T, ctx context.Context, corpusDirectory string) ([]e4bServingCase, []byte) {
	t.Helper()
	oracleBytes, err := os.ReadFile(testutil.FixturePath(t, "e4b_multimodal_golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		Cases []struct {
			Name      string
			PromptIDs []tokenizer.TokenID `json:"prompt_ids"`
			Outputs   []string            `json:"accepted_outputs"`
		}
	}
	if err = json.Unmarshal(oracleBytes, &oracle); err != nil {
		t.Fatal(err)
	}
	imageBytes, err := os.ReadFile(testutil.FixturePath(t, "e4b_vision", "gemma4_mm_image.png"))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(imageBytes)); got != "996fea5cea787f3abb6d1377fc88642ade6bdb8dc9a7b9e6ad909e58687b776e" {
		t.Fatal("image differs from native oracle")
	}
	picture, err := png.Decode(bytes.NewReader(imageBytes))
	if err != nil {
		t.Fatal(err)
	}
	flipped := image.NewRGBA(picture.Bounds())
	for y := picture.Bounds().Min.Y; y < picture.Bounds().Max.Y; y++ {
		for x := picture.Bounds().Min.X; x < picture.Bounds().Max.X; x++ {
			flipped.Set(x, y, picture.At(picture.Bounds().Max.X-1-(x-picture.Bounds().Min.X), y))
		}
	}
	var second bytes.Buffer
	if err := png.Encode(&second, flipped); err != nil {
		t.Fatal(err)
	}
	// FFV1 keeps both native-reference frames lossless; use the actual encoded
	// video route, including decode and ordering, instead of injecting projections.
	ffmpeg, err := media.ResolveFFmpeg("")
	if err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(ctx, ffmpeg, "-v", "error", "-f", "image2pipe", "-framerate", "1", "-i", "pipe:0", "-frames:v", "2", "-c:v", "ffv1", "-f", "matroska", "pipe:1")
	command.Stdin = bytes.NewReader(append(slices.Clone(imageBytes), second.Bytes()...))
	videoBytes, err := command.Output()
	if err != nil {
		t.Fatalf("lossless video fixture: %v", err)
	}
	corpus := filepath.Join(corpusDirectory, "0000.parquet")
	var audioBytes, transcript, recording string
	for _, column := range []string{"bytes", "text", "id"} {
		err := dataset.ReadParquetTextRows(ctx, corpus, column, 1, func(_ uint64, value string) error {
			switch column {
			case "bytes":
				audioBytes = value
			case "text":
				transcript = value
			case "id":
				recording = value
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if recording != "2277-149896-0000" {
		t.Fatalf("wrong corpus row: %q", recording)
	}
	wave, status, err := media.DecodeAudio(ctx, []byte(audioBytes), 16000*30)
	if err != nil {
		t.Fatalf("audio %s: %v", status, err)
	}
	if wave.Format.Channels != 1 || wave.Format.SampleRate != 16000 || len(wave.Samples) != 105440 {
		t.Fatal("audio differs from native oracle geometry")
	}
	wav, err := media.EncodeWAVPCM16(wave.Samples, int(wave.Format.SampleRate))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(wav)); got != "20e10e583b4918e05cea91316f7bde65a345d2756e64abc7f7ee133e2a0547f5" {
		t.Fatal("wave differs from native oracle")
	}
	uri := func(kind string, data []byte) string {
		return "data:" + kind + ";base64," + base64.StdEncoding.EncodeToString(data)
	}
	imageURI, secondURI, audioURI, videoURI := uri("image/png", imageBytes), uri("image/png", second.Bytes()), uri("audio/wav", wav), uri("video/x-matroska", videoBytes)
	transcribe := "Transcribe the speech exactly. Return only the transcript."
	cases := []struct {
		name, question string
		media          []string
	}{
		{"text", "What is 2 plus 2? Answer with the number only.", nil},
		{"image", "What is in this image? One word.", []string{imageURI}},
		{"audio", transcribe, []string{audioURI}},
		{"multi-image", "How many images were supplied? Answer with a number.", []string{imageURI, secondURI}},
		{"video", "Describe the frames in one short sentence.", []string{videoURI}},
		{"mixed", transcribe, []string{imageURI, audioURI}},
	}
	t.Logf("corpus=LibriSpeech-clean-validation row=%s reference=%q; exact native SDPA outputs; 6 modes x 3 routes; unpromoted projection", recording, transcript)
	if len(cases) != len(oracle.Cases) {
		t.Fatal("native case denominator differs")
	}
	result := make([]e4bServingCase, len(cases))
	for i, value := range cases {
		golden := oracle.Cases[i]
		if golden.Name != value.name || len(golden.Outputs) == 0 || len(golden.PromptIDs) == 0 {
			t.Fatal("native case binding is incomplete")
		}
		result[i] = e4bServingCase{value.name, value.question, value.media, golden.PromptIDs, golden.Outputs}
	}
	return result, oracleBytes
}

func encodeE4BServingRequest(t *testing.T, runner *inference.Runner, test e4bServingCase, route string) []byte {
	t.Helper()
	body := map[string]any{"temperature": 0, "stream": false}
	if route == "/v1/chat/completions" {
		body["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
	} else if route == "/v1/responses" {
		body["reasoning"] = map[string]any{"effort": "none"}
	}
	if route == "/completion" {
		body["n_predict"] = 128
		if len(test.media) == 0 {
			formatted, err := runner.FormatChatWithOptions([]inference.ChatMessage{{Role: "user", Content: test.question}}, inference.ChatFormatOptions{AddGenerationPrompt: true})
			if err != nil {
				t.Fatal(err)
			}
			ids, err := runner.TokenizeText(formatted, true, true)
			if err != nil {
				t.Fatal(err)
			}
			body["prompt"] = ids
		} else {
			body["prompt"] = map[string]any{"prompt_string": strings.Repeat("<__media__>", len(test.media)) + test.question, "multimodal_data": test.media}
			if test.name == "mixed" {
				body["prompt"] = map[string]any{"prompt_string": "<|turn>user\n" + strings.Repeat("<__media__>", len(test.media)) + test.question + "<turn|>\n<|turn>model\n", "multimodal_data": test.media}
			}
		}
	} else {
		var content []any
		for _, source := range test.media {
			switch {
			case strings.HasPrefix(source, "data:image/"):
				if route == "/v1/responses" {
					content = append(content, map[string]any{"type": "input_image", "image_url": source})
				} else {
					content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": source}})
				}
			case strings.HasPrefix(source, "data:audio/"):
				content = append(content, map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": strings.SplitN(source, ",", 2)[1], "format": "wav"}})
			case strings.HasPrefix(source, "data:video/"):
				content = append(content, map[string]any{"type": "input_video", "input_video": map[string]any{"data": source, "fps": 1}})
			}
		}
		textType, key, maximum := "text", "messages", "max_tokens"
		if route == "/v1/responses" {
			textType, key, maximum = "input_text", "input", "max_output_tokens"
		}
		content = append(content, map[string]any{"type": textType, "text": test.question})
		body[key], body[maximum] = []any{map[string]any{"role": "user", "content": content}}, 128
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}
