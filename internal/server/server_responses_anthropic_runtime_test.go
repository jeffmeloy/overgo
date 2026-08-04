package server

import (
	"bytes"

	"context"

	"encoding/base64"
	"encoding/json"

	"errors"
	"fmt"

	"image"

	"image/color"
	"image/gif"

	"image/png"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/projector"
	"llamacpp2go/internal/tokenizer"

	"net/http"

	"net/http/httptest"
	"reflect"

	"slices"

	"strings"

	"testing"
)

type mediaToolGenerator struct {
	*fakeGenerator
}

func (generator *mediaToolGenerator) FormatChatWithOptions(
	messages []inference.ChatMessage,
	options inference.ChatFormatOptions,
) (string, error) {
	generator.mu.Lock()
	generator.chatMessages = cloneResponseMessages(messages)
	generator.chatOptions = options
	generator.chatOptions.Tools = append([]inference.ChatTool(nil), options.Tools...)
	generator.mu.Unlock()
	var prompt strings.Builder
	for _, message := range messages {
		prompt.WriteString("<" + message.Role + ">")
		prompt.WriteString(message.Content)
		prompt.WriteString("</" + message.Role + ">")
	}
	return prompt.String(), nil
}

func TestResponsesImageContentPartProjectsPrompt(t *testing.T) {
	generator := &fakeGenerator{}
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	input := image.NewRGBA(image.Rect(0, 0, 1, 1))
	input.SetRGBA(0, 0, color.RGBA{R: 10, G: 20, B: 30, A: 255})
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
	for _, stream := range []bool{false, true} {
		body, err := json.Marshal(map[string]any{
			"input": []any{map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Look "},
					map[string]any{"type": "input_image", "image_url": dataURI, "detail": "auto"},
					map[string]any{"type": "input_text", "text": " now"},
				},
			}},
			"max_output_tokens": 1,
			"stream":            stream,
		})
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body)),
		)
		if response.Code != http.StatusOK {
			t.Fatalf("stream %v status = %d body=%s", stream, response.Code, response.Body.String())
		}
		if stream && !strings.Contains(response.Body.String(), "event: response.completed") {
			t.Fatalf("stream body = %s", response.Body.String())
		}
		if vision.before != "Look " || vision.after != " now" {
			t.Fatalf("projector text = %q, %q", vision.before, vision.after)
		}
		if !slices.Equal(generator.promptIDs, []tokenizer.TokenID{1, 2, 2, 3}) ||
			generator.projectedInputs == nil ||
			len(generator.projectedInputs.EmbeddingOverrides) != 2 ||
			generator.projectedInputs.MultiAxisPositions == nil {
			t.Fatalf("prompt IDs = %v, projected = %+v", generator.promptIDs, generator.projectedInputs)
		}
	}
}

func TestChatAndResponsesEncodedVideoProjection(t *testing.T) {
	videoData := encodedGIFDataURI(t)
	for _, test := range []struct {
		name string
		path string
		body string
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			body: `{"messages":[{"role":"user","content":[` +
				`{"type":"input_video","input_video":{"data":"` + videoData + `","fps":4}},` +
				`{"type":"text","text":"describe"}]}],"max_tokens":1}`,
		},
		{
			name: "responses", path: "/v1/responses",
			body: `{"input":[{"role":"user","content":[` +
				`{"type":"input_video","input_video":{"data":"` + videoData + `","fps":4}},` +
				`{"type":"input_text","text":"describe"}]}],"max_output_tokens":1}`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			generator := &fakeGenerator{}
			video := &fakeVideoProjector{}
			handler, err := New(Config{
				ModelID: "test-model", MaxTokens: 8, DefaultTemperature: 1, DefaultTopP: 1,
				ImageProjector: video, VideoFPS: 2, VideoMaxFrames: 8,
			}, generator)
			if err != nil {
				t.Fatal(err)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(
				response,
				httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body)),
			)
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
			}
			if video.frames != 2 || video.fps != 4 || video.before != "" || video.after != "describe" {
				t.Fatalf("video projection = frames=%d fps=%g before=%q after=%q", video.frames, video.fps, video.before, video.after)
			}
		})
	}
}

func encodedGIFDataURI(t *testing.T) string {
	t.Helper()
	var data bytes.Buffer
	palette := color.Palette{color.Black, color.White}
	frames := &gif.GIF{
		Image: []*image.Paletted{
			image.NewPaletted(image.Rect(0, 0, 2, 2), palette),
			image.NewPaletted(image.Rect(0, 0, 2, 2), palette),
		},
		Delay: []int{1, 1},
	}
	if err := gif.EncodeAll(&data, frames); err != nil {
		t.Fatal(err)
	}
	return "data:video/gif;base64," + base64.StdEncoding.EncodeToString(data.Bytes())
}

func TestResponsesMultipleImagesPreservesContentOrder(t *testing.T) {
	generator := &fakeGenerator{}
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	input := image.NewRGBA(image.Rect(0, 0, 1, 1))
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
	body, err := json.Marshal(map[string]any{
		"input": []any{map[string]any{
			"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "A"},
				map[string]any{"type": "input_image", "image_url": dataURI},
				map[string]any{"type": "input_text", "text": "B"},
				map[string]any{"type": "input_image", "image_url": dataURI},
				map[string]any{"type": "input_text", "text": "C"},
			},
		}}, "max_output_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if vision.images != 2 || !slices.Equal(vision.text, []string{"A", "B", "C"}) {
		t.Fatalf("projector sequence = images %d text %q", vision.images, vision.text)
	}
	if !slices.Equal(generator.promptIDs, []tokenizer.TokenID{1, 2, 2, 3, 4, 4, 5}) ||
		generator.projectedInputs == nil || len(generator.projectedInputs.EmbeddingOverrides) != 4 {
		t.Fatalf("prompt IDs = %v, projected = %+v", generator.promptIDs, generator.projectedInputs)
	}
}

func TestResponsesImageHistoryPreservesTurnPosition(t *testing.T) {
	base := &fakeGenerator{}
	generator := &historyGenerator{fakeGenerator: base}
	vision := &fakeHistoryProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	input := image.NewRGBA(image.Rect(0, 0, 1, 1))
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
	body, err := json.Marshal(map[string]any{
		"input": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_image", "image_url": dataURI},
				map[string]any{"type": "input_text", "text": "inspect"},
			}},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "output_text", "text": "seen"},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "recall"},
			}},
		},
		"max_output_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if vision.images != 1 || !slices.Equal(vision.historyText, []string{
		"<chat><user>",
		"inspect</user><assistant>seen</assistant><user>recall</user><assistant>",
	}) {
		t.Fatalf("projector = images %d text %q", vision.images, vision.historyText)
	}
	if !base.cachePrompt {
		t.Fatal("multimodal Responses prompt cache disabled")
	}
}

func TestResponsesMixedImageAudioPreservesChunkOrder(t *testing.T) {
	base := &fakeGenerator{}
	generator := &historyGenerator{fakeGenerator: base}
	vision := &fakeHistoryProjector{}
	audio := &fakeAudioProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision, AudioProjector: audio,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	var encodedImage bytes.Buffer
	if err := png.Encode(&encodedImage, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	wav := silentPCM16WAV()
	body, err := json.Marshal(map[string]any{
		"input": []any{map[string]any{
			"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "A"},
				map[string]any{"type": "input_image", "image_url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(encodedImage.Bytes())},
				map[string]any{"type": "input_text", "text": "B"},
				map[string]any{"type": "input_audio", "input_audio": map[string]any{
					"data": base64.StdEncoding.EncodeToString(wav), "format": "wav",
				}},
				map[string]any{"type": "input_text", "text": "C"},
			},
		}},
		"max_output_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if !slices.Equal(vision.mediaKinds, []projector.MediaKind{projector.MediaImage, projector.MediaAudio}) ||
		!base.cachePrompt {
		t.Fatalf("media = %v cache=%v", vision.mediaKinds, base.cachePrompt)
	}
}

func TestResponsesMultimodalValidation(t *testing.T) {
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	imagePart := `{"type":"input_image","image_url":"data:image/png;base64,AA=="}`
	for _, test := range []struct {
		path string
		body string
		want string
	}{
		{"/v1/responses", `{"instructions":"brief","input":[{"role":"user","content":[` + imagePart + `]}]}`, "image 0 is unsupported"},
		{"/v1/responses", `{"input":[{"role":"user","content":[` + imagePart + `]}],"tools":[{"type":"function","name":"x","parameters":{}}]}`, "image 0 is unsupported"},
		{"/v1/responses", `{"input":[{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/x.png"}]}]}`, "remote media URLs are disabled"},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body)),
		)
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), test.want) {
			t.Fatalf("%s body %s: status = %d response=%s", test.path, test.body, response.Code, response.Body.String())
		}
	}
}

func TestChatAndResponsesMultimodalFunctionTools(t *testing.T) {
	input := image.NewRGBA(image.Rect(0, 0, 1, 1))
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
	for _, test := range []struct {
		name string
		path string
		body string
		want string
	}{
		{
			name: "chat-buffered", path: "/v1/chat/completions", want: `"finish_reason":"tool_calls"`,
			body: `{"messages":[{"role":"user","content":[` +
				`{"type":"text","text":"inspect"},{"type":"image_url","image_url":{"url":"` + dataURI + `"}}]}],` +
				`"max_tokens":1,"tool_choice":"required","tools":[{"type":"function","function":{` +
				`"name":"weather","parameters":{"type":"object"}}}]}`,
		},
		{
			name: "responses-streaming", path: "/v1/responses", want: "event: response.function_call_arguments.done",
			body: `{"input":[{"role":"user","content":[` +
				`{"type":"input_text","text":"inspect"},{"type":"input_image","image_url":"` + dataURI + `"}]}],` +
				`"max_output_tokens":1,"stream":true,"tool_choice":"required","tools":[{"type":"function",` +
				`"name":"weather","parameters":{"type":"object"}}]}`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			base := &fakeGenerator{pieces: []string{
				`<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`,
			}}
			generator := &mediaToolGenerator{fakeGenerator: base}
			vision := &fakeHistoryProjector{}
			handler, err := New(Config{
				ModelID: "test-model", MaxTokens: 8, DefaultTemperature: 1, DefaultTopP: 1,
				ImageProjector: vision,
			}, generator)
			if err != nil {
				t.Fatal(err)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body)))
			if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), test.want) {
				t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
			}
			if vision.historyRuns != 1 || len(base.chatOptions.Tools) != 1 ||
				base.chatOptions.Tools[0].Function.Name != "weather" {
				t.Fatalf("projection/tools = history=%d options=%+v", vision.historyRuns, base.chatOptions)
			}
		})
	}
}

func TestAnthropicImageToolsAndTokenCounting(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	imageData := base64.StdEncoding.EncodeToString(encoded.Bytes())
	base := &fakeGenerator{pieces: []string{
		`<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`,
	}}
	generator := &mediaToolGenerator{fakeGenerator: base}
	vision := &fakeHistoryProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8, DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	message := `[{"role":"user","content":[` +
		`{"type":"text","text":"before"},{"type":"image","source":{` +
		`"type":"base64","media_type":"image/png","data":"` + imageData + `"}},` +
		`{"type":"text","text":"after"}]}]`
	body := `{"model":"test-model","messages":` + message + `,"max_tokens":1,` +
		`"tool_choice":{"type":"any"},"tools":[{"name":"weather",` +
		`"input_schema":{"type":"object"}}]}`
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"stop_reason":"tool_use"`) {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if vision.historyRuns != 1 || len(base.chatOptions.Tools) != 1 || !base.cachePrompt {
		t.Fatalf(
			"projection/tools = history=%d options=%+v cache=%v",
			vision.historyRuns, base.chatOptions, base.cachePrompt,
		)
	}

	countBody := `{"model":"test-model","messages":` + message + `}`
	count := httptest.NewRecorder()
	handler.ServeHTTP(count, httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(countBody)))
	if count.Code != http.StatusOK || !strings.Contains(count.Body.String(), `"input_tokens":4`) {
		t.Fatalf("count status = %d body=%s", count.Code, count.Body.String())
	}
}

func TestAnthropicThinkingBufferedStreamingImageAndReplay(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	imageData := base64.StdEncoding.EncodeToString(encoded.Bytes())
	newHandler := func() (*Handler, *reasoningGenerator) {
		generator := &reasoningGenerator{fakeGenerator: &fakeGenerator{
			pieces: []string{"check facts</think>", "final answer"},
		}}
		handler, err := New(Config{
			ModelID: "test-model", MaxTokens: 2048,
			DefaultTemperature: 1, DefaultTopP: 1,
			ImageProjector: &fakeHistoryProjector{},
		}, generator)
		if err != nil {
			t.Fatal(err)
		}
		return handler, generator
	}
	thinking := `"thinking":{"type":"enabled","budget_tokens":1024,"display":"summarized"}`
	imageMessage := `[{"role":"user","content":[` +
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + imageData + `"}},` +
		`{"type":"text","text":"inspect"}]}]`

	handler, generator := newHandler()
	body := `{"model":"test-model","max_tokens":1025,` + thinking + `,"messages":` + imageMessage + `}`
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result anthropicResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 2 || result.Content[0].Type != "thinking" ||
		result.Content[0].Thinking != "check facts" ||
		!strings.HasPrefix(result.Content[0].Signature, anthropicThinkingSignaturePrefix) ||
		result.Content[1].Type != "text" || result.Content[1].Text != "final answer" ||
		generator.projectedInputs == nil {
		t.Fatalf("thinking result = %+v projected=%+v", result, generator.projectedInputs)
	}

	content, err := json.Marshal(result.Content)
	if err != nil {
		t.Fatal(err)
	}
	replayMessages := `[{"role":"assistant","content":` + string(content) + `},` +
		`{"role":"user","content":"continue"}]`
	replay := httptest.NewRecorder()
	handler.ServeHTTP(replay, httptest.NewRequest(
		http.MethodPost, "/v1/messages/count_tokens",
		strings.NewReader(`{"model":"test-model",`+thinking+`,"messages":`+replayMessages+`}`),
	))
	if replay.Code != http.StatusOK {
		t.Fatalf("replay status = %d body=%s", replay.Code, replay.Body.String())
	}
	if len(generator.chatMessages) < 2 || generator.chatMessages[0].ReasoningContent != "check facts" {
		t.Fatalf("replayed messages = %+v", generator.chatMessages)
	}

	tampered := strings.Replace(string(content), "check facts", "changed facts", 1)
	badReplay := httptest.NewRecorder()
	handler.ServeHTTP(badReplay, httptest.NewRequest(
		http.MethodPost, "/v1/messages/count_tokens",
		strings.NewReader(`{"model":"test-model","messages":[{"role":"assistant","content":`+
			tampered+`},{"role":"user","content":"continue"}]}`),
	))
	if badReplay.Code != http.StatusBadRequest || !strings.Contains(badReplay.Body.String(), "invalid local thinking signature") {
		t.Fatalf("tamper status = %d body=%s", badReplay.Code, badReplay.Body.String())
	}

	streamHandler, _ := newHandler()
	stream := httptest.NewRecorder()
	streamHandler.ServeHTTP(stream, httptest.NewRequest(
		http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"test-model","max_tokens":1025,"stream":true,`+
			thinking+`,"messages":[{"role":"user","content":"inspect"}]}`),
	))
	if stream.Code != http.StatusOK {
		t.Fatalf("stream status = %d body=%s", stream.Code, stream.Body.String())
	}
	streamBody := stream.Body.String()
	events := []string{
		`"content_block":{"signature":"","thinking":"","type":"thinking"}`,
		`"delta":{"thinking":"check facts","type":"thinking_delta"}`,
		`"delta":{"signature":"` + anthropicThinkingSignaturePrefix,
		`"content_block":{"text":"","type":"text"}`,
		`"delta":{"text":"final answer","type":"text_delta"}`,
	}
	previous := -1
	for _, event := range events {
		index := strings.Index(streamBody, event)
		if index <= previous {
			t.Fatalf("event %q index=%d after=%d body=%s", event, index, previous, streamBody)
		}
		previous = index
	}
}

func TestAnthropicThinkingValidationAndToolPolicy(t *testing.T) {
	for _, test := range []struct {
		name     string
		thinking string
		tools    string
		want     string
	}{
		{name: "small-budget", thinking: `{"type":"enabled","budget_tokens":1023}`, want: "at least 1024"},
		{name: "full-budget", thinking: `{"type":"enabled","budget_tokens":1025}`, want: "less than max_tokens"},
		{name: "adaptive", thinking: `{"type":"adaptive"}`, want: "adaptive thinking is unavailable"},
		{name: "omitted", thinking: `{"type":"enabled","budget_tokens":1024,"display":"omitted"}`, want: "only summarized"},
		{name: "tools", thinking: `{"type":"enabled","budget_tokens":1024}`, tools: `,"tools":[{"name":"weather","input_schema":{"type":"object"}}]`, want: "cannot be combined with tools"},
	} {
		t.Run(test.name, func(t *testing.T) {
			handler, err := New(Config{
				ModelID: "test-model", MaxTokens: 2048,
				DefaultTemperature: 1, DefaultTopP: 1,
			}, &reasoningGenerator{fakeGenerator: &fakeGenerator{}})
			if err != nil {
				t.Fatal(err)
			}
			body := `{"model":"test-model","max_tokens":1025,"thinking":` + test.thinking +
				`,"messages":[{"role":"user","content":"inspect"}]` + test.tools + `}`
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
			if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), test.want) {
				t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func TestBufferedResponsesFunctionCall(t *testing.T) {
	generator := &fakeGenerator{
		pieces: []string{
			`<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`,
		},
	}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(
				`{"input":"weather?","max_output_tokens":1,`+
					`"tool_choice":{"type":"function","name":"weather"},`+
					`"tools":[{"type":"function","name":"weather",`+
					`"description":"forecast","parameters":{"type":"object",`+
					`"properties":{"city":{"type":"string"}}},"strict":true}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result responsesResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Output) != 1 ||
		result.Output[0].Type != "function_call" ||
		result.Output[0].ID == "" ||
		result.Output[0].CallID == "" ||
		result.Output[0].Name != "weather" ||
		result.Output[0].Arguments != `{"city":"Paris"}` {
		t.Fatalf("function response = %+v body=%s", result, response.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatOptions.Tools) != 1 ||
		generator.chatOptions.Tools[0].Function.Name != "weather" ||
		generator.grammarParallel {
		t.Fatalf("formatted tools = %+v", generator.chatOptions.Tools)
	}
}

func TestStreamingResponsesLifecycle(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(`{"input":"hello","max_output_tokens":2,"stream":true}`),
		),
	)
	if response.Code != http.StatusOK ||
		response.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status/headers = %d %#v body=%s", response.Code, response.Header(), response.Body.String())
	}
	body := response.Body.String()
	events := []string{
		"event: response.created",
		"event: response.in_progress",
		"event: response.output_item.added",
		"event: response.content_part.added",
		`"delta":"A"`,
		`"delta":"B"`,
		"event: response.output_text.done",
		"event: response.content_part.done",
		"event: response.output_item.done",
		"event: response.completed",
	}
	previous := -1
	for _, event := range events {
		index := strings.Index(body, event)
		if index < 0 || index <= previous {
			t.Fatalf("event %q index=%d after=%d body=%s", event, index, previous, body)
		}
		previous = index
	}
	if strings.Contains(body, "[DONE]") ||
		!strings.Contains(body, `"output_tokens":2`) ||
		!strings.Contains(body, `"total_tokens":5`) {
		t.Fatalf("stream body = %s", body)
	}
}

func TestStreamingResponsesFunctionCallLifecycle(t *testing.T) {
	generator := &fakeGenerator{
		pieces: []string{
			`<tool_call><function=weather><parameter=city>`,
			`Par`,
			`is</parameter></function></tool_call>`,
		},
	}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(
				`{"input":"weather?","max_output_tokens":3,"stream":true,`+
					`"tool_choice":"required","parallel_tool_calls":false,`+
					`"tools":[{"type":"function",`+
					`"name":"weather","parameters":{"type":"object"}}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	events := []string{
		"event: response.created",
		"event: response.in_progress",
		"event: response.output_item.added",
		`"type":"function_call"`,
		"event: response.function_call_arguments.delta",
		`"delta":"{\"city\":\""`,
		`"delta":"Par"`,
		`"delta":"is\"}"`,
		"event: response.function_call_arguments.done",
		"event: response.output_item.done",
		"event: response.completed",
	}
	previous := -1
	for _, event := range events {
		index := strings.Index(body, event)
		if index < 0 || index <= previous {
			t.Fatalf(
				"event %q index=%d after=%d body=%s",
				event,
				index,
				previous,
				body,
			)
		}
		previous = index
	}
	if strings.Contains(body, "<tool_call>") {
		t.Fatalf("function stream leaked template syntax:\n%s", body)
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if generator.grammarParallel {
		t.Fatal("parallel_tool_calls:false reached the grammar as parallel")
	}
}

func TestStreamingResponsesStoresContinuationHistory(t *testing.T) {
	generator := &fakeGenerator{pieces: []string{"A"}}
	handler := newTestHandler(t, generator)
	stream := httptest.NewRecorder()
	handler.ServeHTTP(
		stream,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(`{"input":"hello","max_output_tokens":1,"stream":true}`),
		),
	)
	if stream.Code != http.StatusOK ||
		!strings.Contains(stream.Body.String(), `"id":"resp_1"`) {
		t.Fatalf("stream status = %d body=%s", stream.Code, stream.Body.String())
	}
	count := httptest.NewRecorder()
	handler.ServeHTTP(
		count,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses/input_tokens",
			strings.NewReader(`{"input":"next","previous_response_id":"resp_1"}`),
		),
	)
	if count.Code != http.StatusOK ||
		!strings.Contains(count.Body.String(), `"object":"response.input_tokens"`) {
		t.Fatalf("count status = %d body=%s", count.Code, count.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	want := []inference.ChatMessage{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "A"},
		{Role: "user", Content: "next"},
	}
	if !reflect.DeepEqual(generator.chatMessages, want) {
		t.Fatalf("continuation count messages = %+v, want %+v", generator.chatMessages, want)
	}
}

func TestResponsesReasoningItemsBufferedAndStreaming(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			generator := &reasoningGenerator{fakeGenerator: &fakeGenerator{
				pieces: []string{"check facts</think>", "final answer"},
			}}
			handler := newTestHandler(t, generator)
			response := httptest.NewRecorder()
			body := fmt.Sprintf(
				`{"input":"question","max_output_tokens":2,"stream":%v,`+
					`"reasoning":{"effort":"high","summary":"auto"}}`,
				stream,
			)
			handler.ServeHTTP(
				response,
				httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)),
			)
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
			}
			if stream {
				want := []string{
					"event: response.reasoning_summary_part.added",
					"event: response.reasoning_summary_text.delta",
					`"delta":"check facts"`,
					"event: response.reasoning_summary_text.done",
					"event: response.reasoning_summary_part.done",
					`"text":"final answer"`,
					"event: response.completed",
				}
				previous := -1
				for _, value := range want {
					index := strings.Index(response.Body.String(), value)
					if index <= previous {
						t.Fatalf("event %q index=%d after=%d body=%s", value, index, previous, response.Body.String())
					}
					previous = index
				}
				return
			}
			var result responsesResponse
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if len(result.Output) != 2 || result.Output[0].Type != "reasoning" ||
				len(result.Output[0].Summary) != 1 || result.Output[0].Summary[0].Text != "check facts" ||
				result.Output[1].Type != "message" || result.Output[1].Content[0].Text != "final answer" {
				t.Fatalf("reasoning response = %+v", result.Output)
			}
		})
	}
}

func TestResponsesReasoningInputItem(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses/input_tokens",
			strings.NewReader(`{"input":[`+
				`{"type":"reasoning","summary":[{"type":"summary_text","text":"prior analysis"}]},`+
				`{"type":"message","role":"user","content":"continue"}]}`),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatMessages) != 2 ||
		generator.chatMessages[0].Role != "assistant" ||
		generator.chatMessages[0].ReasoningContent != "prior analysis" ||
		generator.chatMessages[1].Content != "continue" {
		t.Fatalf("reasoning input messages = %+v", generator.chatMessages)
	}
}

func TestResponsesValidationAndMethod(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{}`,
		`{"input":"hello","max_output_tokens":9}`,
		`{"input":"hello","previous_response_id":"resp_old"}`,
		`{"model":"missing","input":"hello"}`,
		`{"input":"hello","unknown":true}`,
		`{"input":"hello","tools":[{"type":"custom","name":"shell"}]}`,
		`{"input":"hello","tool_choice":{"type":"function","name":"missing"},` +
			`"tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}]}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)),
		)
		if response.Code < 400 {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
	get := httptest.NewRecorder()
	handler.ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/responses", nil))
	if get.Code != http.StatusMethodNotAllowed || get.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status = %d Allow=%q", get.Code, get.Header().Get("Allow"))
	}
}

func TestResponsesInputTokensAuthentication(t *testing.T) {
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{
		"/responses",
		"/v1/responses",
		"/responses/input_tokens",
		"/v1/responses/input_tokens",
		"/v1/messages",
		"/v1/messages/count_tokens",
		"/lora-adapters",
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"input":"hello"}`)),
		)
		if response.Code != http.StatusUnauthorized {
			t.Fatalf("%s status = %d body=%s", path, response.Code, response.Body.String())
		}
	}
	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/v1/health", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("public /v1/health status = %d", health.Code)
	}
}

func TestAnthropicInputTokensTextForms(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"model":"test-model","messages":[{"role":"user","content":"hello"}]}`,
		`{"system":"brief","messages":[{"role":"user","content":[{"type":"text","text":"hello"},{"type":"text","text":" world"}]}]}`,
		`{"system":[{"type":"text","text":"brief"}],"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"again"}],"tools":[]}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/v1/messages/count_tokens",
				strings.NewReader(body),
			),
		)
		if response.Code != http.StatusOK {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
		var result map[string]int
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result["input_tokens"] != 3 {
			t.Fatalf("body %s result = %+v", body, result)
		}
	}
}

func TestAnthropicInputTokensIncludeToolsAndResults(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages/count_tokens",
			strings.NewReader(
				`{"tools":[{"name":"weather","input_schema":{"type":"object"}}],`+
					`"messages":[`+
					`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"weather","input":{"city":"Paris"}}]},`+
					`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"Sunny"}]}`+
					`]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result map[string]int
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["input_tokens"] != 3 {
		t.Fatalf("result = %+v", result)
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatOptions.Tools) != 1 ||
		len(generator.chatMessages) != 2 ||
		len(generator.chatMessages[0].ToolCalls) != 1 ||
		generator.chatMessages[1].Role != "tool" {
		t.Fatalf(
			"formatted options/messages = %+v / %+v",
			generator.chatOptions,
			generator.chatMessages,
		)
	}
}

func TestBufferedAnthropicMessages(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"model":"test-model","system":"brief","max_tokens":2,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result anthropicResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(result.ID, "msg_") ||
		result.Type != "message" ||
		result.Role != "assistant" ||
		len(result.Content) != 1 ||
		result.Content[0].Type != "text" ||
		result.Content[0].Text != "AB" ||
		result.StopReason != "max_tokens" ||
		result.StopSequence != nil ||
		result.Usage.InputTokens != 3 ||
		result.Usage.OutputTokens != 2 {
		t.Fatalf("response = %+v body=%s", result, response.Body.String())
	}
}

func TestBufferedAnthropicToolUse(t *testing.T) {
	generator := &fakeGenerator{
		pieces: []string{
			`<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`,
		},
	}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"max_tokens":1,"tool_choice":{"type":"any"},`+
					`"tools":[{"name":"weather","description":"forecast",`+
					`"input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],`+
					`"messages":[{"role":"user","content":"weather?"}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result anthropicResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "tool_use" ||
		len(result.Content) != 1 ||
		result.Content[0].Type != "tool_use" ||
		result.Content[0].ID == "" ||
		result.Content[0].Name != "weather" ||
		string(result.Content[0].Input) != `{"city":"Paris"}` {
		t.Fatalf("tool response = %+v body=%s", result, response.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatOptions.Tools) != 1 ||
		generator.chatOptions.Tools[0].Function.Name != "weather" {
		t.Fatalf("formatted tools = %+v", generator.chatOptions.Tools)
	}
}

func TestAnthropicMessagesStopAndValidation(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	stopped := httptest.NewRecorder()
	handler.ServeHTTP(
		stopped,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"max_tokens":2,"stop_sequences":["AB"],"messages":[{"role":"user","content":"hello"}]}`,
			),
		),
	)
	if stopped.Code != http.StatusOK {
		t.Fatalf("stop status = %d body=%s", stopped.Code, stopped.Body.String())
	}
	var result anthropicResponse
	if err := json.Unmarshal(stopped.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "end_turn" ||
		result.StopSequence == nil ||
		*result.StopSequence != "AB" ||
		len(result.Content) != 0 {
		t.Fatalf("stopped response = %+v", result)
	}
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hello"}]}`,
		`{"max_tokens":9,"messages":[{"role":"user","content":"hello"}]}`,
		`{"max_tokens":1,"tools":[{"name":"x"}],"messages":[{"role":"user","content":"hello"}]}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)),
		)
		if response.Code < 400 {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
}

func TestStreamingAnthropicMessagesLifecycle(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"max_tokens":2,"stream":true,"messages":[{"role":"user","content":"hello"}]}`,
			),
		),
	)
	if response.Code != http.StatusOK ||
		response.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status/headers = %d %#v body=%s", response.Code, response.Header(), response.Body.String())
	}
	body := response.Body.String()
	events := []string{
		"event: message_start",
		"event: content_block_start",
		`"text":"A"`,
		`"text":"B"`,
		"event: content_block_stop",
		"event: message_delta",
		"event: message_stop",
	}
	previous := -1
	for _, event := range events {
		index := strings.Index(body, event)
		if index < 0 || index <= previous {
			t.Fatalf("event %q index=%d after=%d body=%s", event, index, previous, body)
		}
		previous = index
	}
	if strings.Contains(body, "timings") ||
		!strings.Contains(body, `"stop_reason":"max_tokens"`) ||
		!strings.Contains(body, `"output_tokens":2`) {
		t.Fatalf("stream body = %s", body)
	}
}

func TestStreamingAnthropicToolUseLifecycle(t *testing.T) {
	generator := &fakeGenerator{
		pieces: []string{
			`<tool_call><function=weather><parameter=city>`,
			`Par`,
			`is</parameter></function></tool_call>`,
		},
	}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"max_tokens":3,"stream":true,"tool_choice":{"type":"tool","name":"weather",`+
					`"disable_parallel_tool_use":true},`+
					`"tools":[{"name":"weather","input_schema":{"type":"object"}}],`+
					`"messages":[{"role":"user","content":"weather?"}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	for _, fragment := range []string{
		"event: message_start",
		`"id":"toolu_`,
		`"type":"tool_use"`,
		`"name":"weather"`,
		`"type":"input_json_delta"`,
		`"partial_json":"{\"city\":\""`,
		`"partial_json":"Par"`,
		`"partial_json":"is\"}"`,
		`"stop_reason":"tool_use"`,
		"event: message_stop",
	} {
		if !strings.Contains(body, fragment) {
			t.Fatalf("tool stream lacks %q:\n%s", fragment, body)
		}
	}
	if strings.Contains(body, "<tool_call>") {
		t.Fatalf("tool stream leaked template syntax:\n%s", body)
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if generator.grammarParallel {
		t.Fatal("disable_parallel_tool_use reached the grammar as parallel")
	}
}

func TestParseAnthropicToolHistory(t *testing.T) {
	messages, err := parseAnthropicMessages(
		nil,
		json.RawMessage(`[
			{"role":"assistant","content":[
				{"type":"text","text":"Checking."},
				{"type":"tool_use","id":"toolu_1","name":"weather","input":{"city":"Paris"}}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"Sunny"}],"is_error":true},
				{"type":"text","text":"Thanks"}
			]}
		]`),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 3 ||
		messages[0].Content != "Checking." ||
		len(messages[0].ToolCalls) != 1 ||
		messages[0].ToolCalls[0].ID != "toolu_1" ||
		messages[1].Role != "tool" ||
		messages[1].ToolCallID != "toolu_1" ||
		!messages[1].ToolResultError ||
		messages[1].Content != "Sunny" ||
		messages[2].Role != "user" ||
		messages[2].Content != "Thanks" {
		t.Fatalf("parsed messages = %+v", messages)
	}
}

func TestAnthropicInputTokensValidationAndMethod(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{}`,
		`{"messages":[]}`,
		`{"model":"missing","messages":[{"role":"user","content":"hi"}]}`,
		`{"messages":[{"role":"tool","content":"hi"}]}`,
		`{"messages":[{"role":"user","content":[{"type":"image","source":{}}]}]}`,
		`{"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"x"}]}`,
		`{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled"}}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/v1/messages/count_tokens",
				strings.NewReader(body),
			),
		)
		if response.Code < 400 {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
	get := httptest.NewRecorder()
	handler.ServeHTTP(
		get,
		httptest.NewRequest(http.MethodGet, "/v1/messages/count_tokens", nil),
	)
	if get.Code != http.StatusMethodNotAllowed || get.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status = %d Allow=%q", get.Code, get.Header().Get("Allow"))
	}
}

func TestChatInputTokensValidationAuthenticationAndMethod(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"messages":[]}`,
		`{"model":"missing","messages":[{"role":"user","content":"hi"}]}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/chat/completions/input_tokens",
				strings.NewReader(body),
			),
		)
		if response.Code < 400 {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
	get := httptest.NewRecorder()
	handler.ServeHTTP(
		get,
		httptest.NewRequest(http.MethodGet, "/chat/completions/input_tokens", nil),
	)
	if get.Code != http.StatusMethodNotAllowed ||
		get.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status = %d Allow=%q", get.Code, get.Header().Get("Allow"))
	}

	protected, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	unauthorized := httptest.NewRecorder()
	protected.ServeHTTP(
		unauthorized,
		httptest.NewRequest(
			http.MethodPost,
			"/chat/completions/input_tokens",
			strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`),
		),
	)
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d body=%s", unauthorized.Code, unauthorized.Body.String())
	}
}

func TestCompletionRejectsInvalidPromptShapesAndBatchBounds(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	largeBatch := make([]string, 65)
	for index := range largeBatch {
		largeBatch[index] = "prompt"
	}
	largeJSON, err := json.Marshal(map[string]any{"prompt": largeBatch})
	if err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{
		`{}`,
		`{"prompt":[]}`,
		`{"prompt":[64]}`,
		`{"prompt":[[5],[]]}`,
		string(largeJSON),
	} {
		request := httptest.NewRequest(
			http.MethodPost,
			"/v1/completions",
			strings.NewReader(body),
		)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
}

func TestApplyTemplate(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/apply-template",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result map[string]string
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["prompt"] != "formatted-chat" {
		t.Fatalf("apply-template response = %#v", result)
	}
}

func TestApplyTemplateSuppliesToolsAndTemplateOptions(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	body := `{
		"messages":[
			{"role":"user","content":"weather?"},
			{"role":"assistant","content":null,"tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}
			]},
			{"role":"tool","tool_call_id":"call_1","content":"sunny"}
		],
		"tools":[{"type":"function","function":{
			"name":"weather",
			"description":"Get weather",
			"parameters":{"type":"object","properties":{"city":{"type":"string"}}}
		}}],
		"add_generation_prompt":false,
		"chat_template_kwargs":{"enable_thinking":false}
	}`
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(http.MethodPost, "/apply-template", strings.NewReader(body)),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result map[string]string
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["prompt"] != "formatted-chat-tools" {
		t.Fatalf("apply-template response = %#v", result)
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatOptions.Tools) != 1 ||
		generator.chatOptions.Tools[0].Function.Name != "weather" ||
		generator.chatOptions.AddGenerationPrompt ||
		generator.chatOptions.EnableThinking ||
		len(generator.chatMessages) != 3 ||
		len(generator.chatMessages[1].ToolCalls) != 1 {
		t.Fatalf(
			"messages=%+v options=%+v",
			generator.chatMessages,
			generator.chatOptions,
		)
	}
}

func TestApplyTemplateRejectsMethodAndInvalidMessages(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := httptest.NewRequest(http.MethodGet, "/apply-template", nil)
	getResponse := httptest.NewRecorder()
	handler.ServeHTTP(getResponse, get)
	if getResponse.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d", getResponse.Code)
	}
	post := httptest.NewRequest(
		http.MethodPost,
		"/apply-template",
		strings.NewReader(`{"messages":[]}`),
	)
	postResponse := httptest.NewRecorder()
	handler.ServeHTTP(postResponse, post)
	if postResponse.Code != http.StatusBadRequest {
		t.Fatalf("empty messages status = %d body=%s", postResponse.Code, postResponse.Body.String())
	}
}

func TestTokenizeAndDetokenizeEndpoints(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	tokenize := httptest.NewRequest(
		http.MethodPost,
		"/tokenize",
		strings.NewReader(`{"content":"hello","add_special":true,"parse_special":true}`),
	)
	tokenizeResponse := httptest.NewRecorder()
	handler.ServeHTTP(tokenizeResponse, tokenize)
	if tokenizeResponse.Code != http.StatusOK ||
		!strings.Contains(tokenizeResponse.Body.String(), `"tokens":[1,10,2]`) {
		t.Fatalf("tokenize status/body = %d %s", tokenizeResponse.Code, tokenizeResponse.Body.String())
	}
	detokenize := httptest.NewRequest(
		http.MethodPost,
		"/detokenize",
		strings.NewReader(`{"tokens":[1,10,2]}`),
	)
	detokenizeResponse := httptest.NewRecorder()
	handler.ServeHTTP(detokenizeResponse, detokenize)
	var detokenizeResult map[string]string
	detokenizeErr := json.Unmarshal(detokenizeResponse.Body.Bytes(), &detokenizeResult)
	if detokenizeResponse.Code != http.StatusOK ||
		detokenizeErr != nil ||
		detokenizeResult["content"] != "<1><10><2>" {
		t.Fatalf(
			"detokenize status/body = %d %s",
			detokenizeResponse.Code,
			detokenizeResponse.Body.String(),
		)
	}
}

func TestTokenizeMixedContentAndPinnedDefaults(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, test := range []struct {
		name string
		body string
		want string
	}{
		{
			name: "strings and token",
			body: `{"content":["first",5,"second"],"add_special":true}`,
			want: `"tokens":[1,10,2,5,10,2]`,
		},
		{
			name: "leading token suppresses special",
			body: `{"content":[5,"second"],"add_special":true}`,
			want: `"tokens":[5,10,2]`,
		},
		{
			name: "parse special override",
			body: `{"content":"plain","parse_special":false}`,
			want: `"tokens":[10]`,
		},
		{
			name: "missing content",
			body: `{}`,
			want: `"tokens":[]`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/tokenize", strings.NewReader(test.body))
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), test.want) {
				t.Fatalf("status/body = %d %s, want %s", response.Code, response.Body.String(), test.want)
			}
		})
	}
}

func TestTokenizeWithPiecesPreservesInvalidUTF8AsBytes(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/tokenize",
		strings.NewReader(`{"content":[5,63],"with_pieces":true}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result struct {
		Tokens []struct {
			ID    int             `json:"id"`
			Piece json.RawMessage `json:"piece"`
		} `json:"tokens"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Tokens) != 2 {
		t.Fatalf("pieces = %s", response.Body.String())
	}
	var firstPiece string
	if err := json.Unmarshal(result.Tokens[0].Piece, &firstPiece); err != nil {
		t.Fatal(err)
	}
	if result.Tokens[0].ID != 5 ||
		firstPiece != "<5>" ||
		result.Tokens[1].ID != 63 ||
		string(result.Tokens[1].Piece) != `[195]` {
		t.Fatalf("pieces = %s", response.Body.String())
	}
}

func TestTokenizeRejectsInvalidMixedContent(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"content":[64]}`,
		`{"content":[1.5]}`,
		`{"content":{"text":"bad"}}`,
	} {
		request := httptest.NewRequest(http.MethodPost, "/tokenize", strings.NewReader(body))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s: status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
}

func TestDetokenizeRejectsOutOfRangeToken(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/detokenize",
		strings.NewReader(`{"tokens":[64]}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
}

func TestChatCompletionMultipleChoices(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"n":2}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result chatResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Choices) != 2 ||
		result.Choices[0].Index != 0 ||
		result.Choices[1].Index != 1 {
		t.Fatalf("chat choices = %+v", result.Choices)
	}
	if result.Usage.CompletionTokens != 2 {
		t.Fatalf("chat usage = %+v", result.Usage)
	}
}

func TestStreamingChatCompletion(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"stream":true}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	for _, fragment := range []string{
		`"role":"assistant"`,
		`"content":"A"`,
		`"content":"B"`,
		`"finish_reason":"length"`,
		"data: [DONE]",
	} {
		if !strings.Contains(body, fragment) {
			t.Fatalf("chat stream lacks %q: %s", fragment, body)
		}
	}
}

func TestRejectsInvalidRequest(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"prompt":""}`,
		`{"prompt":"x","max_tokens":9}`,
		`{"prompt":"x","unknown":true}`,
		`{"prompt":"x","temperature":-1}`,
	} {
		request := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s status = %d, want 400", body, response.Code)
		}
	}
}

func TestAdmissionControl(t *testing.T) {
	blocking := &fakeGenerator{started: make(chan struct{}), release: make(chan struct{})}
	handler := newTestHandler(t, blocking)
	firstDone := make(chan struct{})
	go func() {
		defer close(firstDone)
		request := httptest.NewRequest(
			http.MethodPost,
			"/v1/completions",
			strings.NewReader(`{"prompt":"first","max_tokens":1}`),
		)
		handler.ServeHTTP(httptest.NewRecorder(), request)
	}()
	<-blocking.started
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"second","max_tokens":1}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", response.Code)
	}
	close(blocking.release)
	<-firstDone
}

func TestCancellationPropagates(t *testing.T) {
	blocking := &fakeGenerator{started: make(chan struct{}), release: make(chan struct{})}
	handler := newTestHandler(t, blocking)
	ctx, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"cancel","max_tokens":1}`),
	).WithContext(ctx)
	response := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		handler.ServeHTTP(response, request)
	}()
	<-blocking.started
	cancel()
	<-done
	if response.Code != http.StatusRequestTimeout {
		t.Fatalf("status = %d, want 408", response.Code)
	}
	var envelope apiErrorEnvelope
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(ctx.Err(), context.Canceled) || envelope.Error.Type != "request_cancelled" {
		t.Fatalf("error = %+v, context = %v", envelope, ctx.Err())
	}
}
