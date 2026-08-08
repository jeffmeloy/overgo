package recipe

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	LifecycleVersion   = uint16(1)
	LifecycleMediaType = "application/vnd.overgo.recipe-lifecycle+json"
	LifecycleSchema    = "overgo.recipe-lifecycle.v1"
)

var lifecycleContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: LifecycleMediaType, Schema: LifecycleSchema,
}

var lifecycleCodec = artifact.DocumentCodec[LifecycleEvent]{
	Name: "recipe lifecycle", Contract: lifecycleContract,
	Decode: func(data []byte, value *LifecycleEvent) error {
		var body lifecycleBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = LifecycleEvent{
			Version: body.Version, Recipe: body.Recipe, Model: body.Model, Task: body.Task,
			From: body.From, To: body.To, PreviousEvent: body.PreviousEvent,
			Supersedes: body.Supersedes, Evidence: body.Evidence,
		}
		return nil
	},
	Encode: lifecycleContent, Canonicalize: canonicalizeEvent,
	Clone: func(value LifecycleEvent) LifecycleEvent {
		value.PreviousEvent = artifact.CloneID(value.PreviousEvent)
		value.Supersedes = artifact.CloneID(value.Supersedes)
		value.Evidence = slices.Clone(value.Evidence)
		return value
	},
	Identity:    func(value LifecycleEvent) artifact.ID { return value.ID },
	SetIdentity: func(value *LifecycleEvent, id artifact.ID) { value.ID = id },
}

func LifecycleDocumentContract() artifact.DocumentContract { return lifecycleContract }

type Status string

const (
	StatusCandidate  Status = "candidate"
	StatusValidated  Status = "validated"
	StatusActive     Status = "active"
	StatusRefused    Status = "refused"
	StatusSuperseded Status = "superseded"
)

type LifecycleEvent struct {
	Version       uint16        `json:"version"`
	ID            artifact.ID   `json:"id"`
	Recipe        artifact.ID   `json:"recipe"`
	Model         artifact.ID   `json:"model"`
	Task          Task          `json:"task"`
	From          Status        `json:"from,omitempty"`
	To            Status        `json:"to"`
	PreviousEvent *artifact.ID  `json:"previous_event,omitempty"`
	Supersedes    *artifact.ID  `json:"supersedes,omitempty"`
	Evidence      []artifact.ID `json:"evidence,omitempty"`
}

type lifecycleBody struct {
	Version       uint16        `json:"version"`
	Recipe        artifact.ID   `json:"recipe"`
	Model         artifact.ID   `json:"model"`
	Task          Task          `json:"task"`
	From          Status        `json:"from,omitempty"`
	To            Status        `json:"to"`
	PreviousEvent *artifact.ID  `json:"previous_event,omitempty"`
	Supersedes    *artifact.ID  `json:"supersedes,omitempty"`
	Evidence      []artifact.ID `json:"evidence,omitempty"`
}

func NewLifecycleEvent(definition Definition, from, to Status, previousEvent, supersedes *artifact.ID, evidence []artifact.ID) (LifecycleEvent, error) {
	if err := definition.ValidateIdentity(); err != nil {
		return LifecycleEvent{}, err
	}
	return lifecycleCodec.New(LifecycleEvent{
		Version: LifecycleVersion, Recipe: definition.ID, Model: definition.Model, Task: definition.Task,
		From: from, To: to, PreviousEvent: artifact.CloneID(previousEvent), Supersedes: artifact.CloneID(supersedes),
		Evidence: slices.Clone(evidence),
	})
}

func ParseLifecycleEvent(content []byte) (LifecycleEvent, error) {
	return lifecycleCodec.Parse(content)
}

func (e LifecycleEvent) Content() (artifact.Content, error) {
	return lifecycleCodec.Content(e)
}

func (e LifecycleEvent) Validate() error {
	return lifecycleCodec.ValidateIdentity(e)
}

func canonicalizeEvent(event *LifecycleEvent) error {
	if event == nil || event.Version != LifecycleVersion {
		return errors.New("recipe: invalid lifecycle version")
	}
	if event.Recipe.Kind() != artifact.KindRecipe || event.Model.Kind() != artifact.KindModel {
		return errors.New("recipe: invalid lifecycle subject")
	}
	if event.PreviousEvent != nil && event.PreviousEvent.Kind() != artifact.KindEvidence {
		return errors.New("recipe: invalid previous lifecycle event")
	}
	if event.Supersedes != nil && event.Supersedes.Kind() != artifact.KindRecipe {
		return errors.New("recipe: invalid superseded recipe")
	}
	if err := validateTask(event.Task); err != nil {
		return err
	}
	if !validTransition(event.From, event.To) {
		return fmt.Errorf("recipe: invalid lifecycle transition %q -> %q", event.From, event.To)
	}
	if event.From == "" && event.PreviousEvent != nil || event.From != "" && event.PreviousEvent == nil {
		return errors.New("recipe: lifecycle previous event mismatch")
	}
	if event.To == StatusActive && len(event.Evidence) == 0 {
		return errors.New("recipe: activation requires evidence")
	}
	for _, id := range event.Evidence {
		if !id.Valid() {
			return errors.New("recipe: invalid lifecycle evidence")
		}
	}
	sort.Slice(event.Evidence, func(i, j int) bool { return event.Evidence[i].String() < event.Evidence[j].String() })
	event.Evidence = slices.Compact(event.Evidence)
	return nil
}

func validTransition(from, to Status) bool {
	switch from {
	case "":
		return to == StatusCandidate
	case StatusCandidate:
		return to == StatusValidated || to == StatusRefused
	case StatusValidated:
		return to == StatusActive || to == StatusRefused
	case StatusActive:
		return to == StatusSuperseded
	default:
		return false
	}
}

func lifecycleContent(event LifecycleEvent) ([]byte, error) {
	return json.Marshal(lifecycleBody{
		Version: event.Version, Recipe: event.Recipe, Model: event.Model, Task: event.Task,
		From: event.From, To: event.To, PreviousEvent: event.PreviousEvent,
		Supersedes: event.Supersedes, Evidence: event.Evidence,
	})
}
