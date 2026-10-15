package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"maps"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/strictjson"
)

// MediaRuntime names the loaded answering model and its admitted recipes.
type MediaRuntime struct {
	Model      artifact.ID `json:"model"`
	Recipe     artifact.ID `json:"recipe"`
	Projection artifact.ID `json:"projection,omitzero"`
}

// MediaDecode names the decoder and the frame sampling consumed by a case.
type MediaDecode struct {
	Format string      `json:"format"`
	FPS    float64     `json:"fps,omitzero"`
	Frames int         `json:"frames,omitzero"`
	Tool   artifact.ID `json:"tool,omitzero"`
}

// MediaInput separates consumed bytes from a generated payload's request and producer.
type MediaInput struct {
	Content  artifact.ID  `json:"content,omitzero"`
	Runtime  MediaRuntime `json:"runtime,omitzero"`
	Decode   MediaDecode  `json:"decode"`
	Request  artifact.ID  `json:"request,omitzero"`
	Producer artifact.ID  `json:"producer,omitzero"`
}

// Reference names the bytes and execution that determine an answer.
// A second producer of identical bytes does not require another identical judgment.
func (input MediaInput) Reference() (string, error) {
	input.Request, input.Producer = artifact.ID{}, artifact.ID{}
	id, err := artifact.JSONID(artifact.KindFile, input)
	return id.DigestHex(), err
}

func (input MediaInput) valid() bool {
	generated := input.Request.Kind() == artifact.KindFile && input.Producer.Kind() == artifact.KindRun && input.Content.Kind() == artifact.KindOutput
	fixed := !input.Request.Valid() && !input.Producer.Valid() && input.Content.Kind() == artifact.KindFile
	image := input.Decode.Format == "image" && input.Decode.FPS == 0 && input.Decode.Frames == 0 && !input.Decode.Tool.Valid()
	video := checked.PositiveFinite64(input.Decode.FPS) && input.Decode.Frames > 0 &&
		(input.Decode.Format == "gif" && !input.Decode.Tool.Valid() || input.Decode.Format == "ffmpeg" && input.Decode.Tool.Kind() == artifact.KindFile)
	return (generated || fixed) &&
		input.Runtime.Model.Kind() == artifact.KindModel && input.Runtime.Recipe.Kind() == artifact.KindRecipe &&
		(!input.Runtime.Projection.Valid() || input.Runtime.Projection.Kind() == artifact.KindRecipe) &&
		(image || video)
}

type mediaAuthority struct {
	Method artifact.ID  `json:"method"`
	Inputs []MediaInput `json:"inputs"`
}

// CompileSuite resolves media before compilation, comparison or answer lookup.
// Text-only runtimes preserve the ordinary suite compiler.
func (campaign *Campaign) CompileSuite(ctx context.Context, data []byte) (CompiledSuite, error) {
	if campaign == nil || ctx == nil {
		return CompiledSuite{}, errors.New("evaluation: campaign is absent")
	}
	authorities := campaign.Authorities()
	resolver, ok := campaign.runtime.(interface {
		ResolveMedia(context.Context, MediaKind, string) (MediaInput, error)
	})
	if !ok {
		return CompileSuite(data, authorities)
	}
	var document map[string]json.RawMessage
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return CompiledSuite{}, err
	}
	var cases []map[string]json.RawMessage
	if err := json.Unmarshal(document["cases"], &cases); err != nil {
		return CompiledSuite{}, err
	}
	methodCases := make([]map[string]json.RawMessage, len(cases))
	var inputs []MediaInput
	resolved := map[struct {
		kind MediaKind
		name string
	}]MediaInput{}
	for index, testCase := range cases {
		if err := ctx.Err(); err != nil {
			return CompiledSuite{}, err
		}
		methodCases[index] = maps.Clone(testCase)
		var prompt string
		if raw, exists := testCase["prompt"]; exists {
			if err := json.Unmarshal(raw, &prompt); err != nil {
				return CompiledSuite{}, err
			}
		}
		kind, name, question, named := ParseMediaPrompt(prompt)
		if !named {
			continue
		}
		key := struct {
			kind MediaKind
			name string
		}{kind, name}
		input, known := resolved[key]
		if !known {
			var err error
			input, err = resolver.ResolveMedia(ctx, kind, name)
			if err != nil {
				return CompiledSuite{}, err
			}
			resolved[key] = input
		}
		if !input.valid() || (kind == MediaImage) != (input.Decode.Format == "image") {
			return CompiledSuite{}, errors.New("evaluation: media input authority is incomplete")
		}
		inputs = append(inputs, input)
		reference, err := input.Reference()
		if err != nil {
			return CompiledSuite{}, err
		}
		testCase["prompt"], err = json.Marshal(mediaPrompt(kind, reference, question))
		if err != nil {
			return CompiledSuite{}, err
		}
		method := input
		method.Producer = artifact.ID{}
		if method.Request.Valid() {
			method.Content = artifact.ID{}
		} else {
			method.Runtime = MediaRuntime{}
		}
		methodID, err := artifact.JSONID(artifact.KindProfile, method)
		if err != nil {
			return CompiledSuite{}, err
		}
		methodCases[index]["prompt"], err = json.Marshal(mediaPrompt(kind, methodID.DigestHex(), question))
		if err != nil {
			return CompiledSuite{}, err
		}
		if input.Request.Valid() {
			// The session's admitted judge replaces a suite's descriptive label.
			judge, err := artifact.JSONID(artifact.KindProfile, input.Runtime)
			if err != nil {
				return CompiledSuite{}, err
			}
			document["judge"], err = json.Marshal(judge.String())
			if err != nil {
				return CompiledSuite{}, err
			}
		}
	}
	if len(inputs) == 0 {
		return CompileSuite(data, authorities)
	}
	document["cases"], _ = json.Marshal(methodCases)
	methodData, err := json.Marshal(document)
	if err != nil {
		return CompiledSuite{}, err
	}
	method, err := CompileSuite(methodData, authorities)
	if err != nil {
		return CompiledSuite{}, err
	}
	authorities.media = &mediaAuthority{Method: method.plan.body.CaseProfile, Inputs: inputs}
	authorities.mediaProfile = method.plan.authorities[0]
	document["cases"], _ = json.Marshal(cases)
	bound, err := json.Marshal(document)
	if err != nil {
		return CompiledSuite{}, err
	}
	return CompileSuite(bound, authorities)
}

// namesMedia identifies structured media prompts in a retained case profile.
func namesMedia(data []byte) (bool, error) {
	var document struct {
		Cases []struct {
			Prompt string `json:"prompt"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return false, err
	}
	for _, testCase := range document.Cases {
		if _, _, _, named := ParseMediaPrompt(testCase.Prompt); named {
			return true, nil
		}
	}
	return false, nil
}
