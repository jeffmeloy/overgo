// Package controllertrain owns workflow-controller data and promotion evidence.
package controllertrain

import (
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"slices"
	"strings"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/workflowrecipe"
)

const (
	CorpusVersion       uint16 = 1
	CorpusMediaType            = "application/vnd.overgo.controller-corpus+json"
	CorpusSchema               = "overgo/controller-corpus/v1"
	SplitMediaType             = "application/vnd.overgo.controller-split+json"
	SplitSchema                = "overgo/controller-split/v1"
	trainingPromptWidth        = 56
	trainingLabelWidth         = 64
)

type Scope string

const (
	ScopeComponent Scope = "component"
	ScopeWorkflow  Scope = "workflow"
)

type Modality string

const (
	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
	ModalityVideo Modality = "video"
)

// GitSource: immutable corpus provenance.
type GitSource struct {
	Commit string `json:"commit"`
	Path   string `json:"path"`
}

// Action: typed workflow selection target.
type Action struct {
	Scope    Scope           `json:"scope"`
	Task     recipe.Task     `json:"task"`
	Modality Modality        `json:"modality"`
	Module   recipe.ModuleID `json:"module"`
}

// Record: one explicit controller example.
type Record struct {
	ID     string    `json:"id"`
	Group  string    `json:"group"`
	Source GitSource `json:"source"`
	Prompt string    `json:"prompt"`
	Action Action    `json:"action"`
}

type Spec struct {
	Train   []Record
	Holdout []Record
}

// Corpus: immutable external-holdout controller dataset.
type Corpus struct {
	dataset      artifact.ID
	split        artifact.ID
	train        []Record
	holdout      []Record
	actions      []Action
	labels       map[string]string
	content      artifact.Content
	splitContent artifact.Content
}

type corpusDocument struct {
	Version uint16   `json:"version"`
	Train   []Record `json:"train"`
	Holdout []Record `json:"holdout"`
}

type splitDocument struct {
	Version uint16      `json:"version"`
	Dataset artifact.ID `json:"dataset"`
	Train   []string    `json:"train"`
	Holdout []string    `json:"holdout"`
}

var (
	corpusContract = artifact.DocumentContract{Kind: artifact.KindDataset, MediaType: CorpusMediaType, Schema: CorpusSchema}
	splitContract  = artifact.DocumentContract{Kind: artifact.KindDatasetShard, MediaType: SplitMediaType, Schema: SplitSchema}
)

func Compile(spec Spec) (Corpus, error) {
	train, holdout := slices.Clone(spec.Train), slices.Clone(spec.Holdout)
	if len(train) < 3 || len(holdout) == 0 {
		return Corpus{}, errors.New("controller training: train and holdout records required")
	}
	if err := validateRecords(train, holdout); err != nil {
		return Corpus{}, err
	}
	actions := collectActions(train, holdout)
	if err := validateCoverage(actions); err != nil {
		return Corpus{}, err
	}
	labels := make(map[string]string, len(actions))
	for _, action := range actions {
		labels[actionKey(action)] = actionKey(action)
	}
	documentBytes, err := json.Marshal(corpusDocument{Version: CorpusVersion, Train: train, Holdout: holdout})
	if err != nil {
		return Corpus{}, err
	}
	content, err := corpusContract.ContentBytes(documentBytes)
	if err != nil {
		return Corpus{}, err
	}
	splitBytes, err := json.Marshal(splitDocument{
		Version: CorpusVersion, Dataset: content.Descriptor.ID,
		Train: recordIDs(train), Holdout: recordIDs(holdout),
	})
	if err != nil {
		return Corpus{}, err
	}
	splitContent, err := splitContract.ContentBytes(splitBytes)
	if err != nil {
		return Corpus{}, err
	}
	corpus := Corpus{
		dataset: content.Descriptor.ID, split: splitContent.Descriptor.ID,
		train: train, holdout: holdout, actions: actions, labels: labels,
		content: content, splitContent: splitContent,
	}
	if err := corpus.validateTokenizerCoverage(); err != nil {
		return Corpus{}, err
	}
	return corpus, nil
}

func (c Corpus) Dataset() artifact.ID { return c.dataset }
func (c Corpus) Split() artifact.ID   { return c.split }
func (c Corpus) Actions() []Action    { return slices.Clone(c.actions) }
func (c Corpus) Holdout() []Record    { return slices.Clone(c.holdout) }

func (c Corpus) TrainingDocuments() []string {
	result := make([]string, len(c.train))
	for index, record := range c.train {
		result[index] = c.document(record.Prompt, record.Action)
	}
	return result
}

func (c Corpus) CandidateDocument(prompt string, action Action) (string, error) {
	if _, ok := c.labels[actionKey(action)]; !ok {
		return "", errors.New("controller training: action absent from corpus")
	}
	return c.document(prompt, action), nil
}

func (c Corpus) PublicationBatch(key string) (artifact.Batch, error) {
	return artifact.NewDocumentBatch(key,
		[]artifact.Content{c.content, c.splitContent},
		[]artifact.Lineage{{Child: c.split, Parent: c.dataset, Relation: artifact.RelationDerivedFrom}}, nil,
	)
}

func (c Corpus) document(prompt string, action Action) string {
	label := c.labels[actionKey(action)]
	return fmt.Sprintf("request: %s\naction: %-*s", prompt, trainingLabelWidth, label)
}

func (c Corpus) validateTokenizerCoverage() error {
	available := map[rune]struct{}{}
	for _, document := range c.TrainingDocuments() {
		for _, character := range document {
			available[character] = struct{}{}
		}
	}
	for _, record := range c.holdout {
		for _, action := range c.actions {
			document := c.document(record.Prompt, action)
			for _, character := range document {
				if _, ok := available[character]; !ok {
					return fmt.Errorf("controller training: holdout character %q absent from training corpus", character)
				}
			}
		}
	}
	return nil
}

func validateRecords(train, holdout []Record) error {
	ids, trainGroups := map[string]struct{}{}, map[string]struct{}{}
	for splitIndex, records := range [][]Record{train, holdout} {
		for _, record := range records {
			if !validToken(record.ID) || !validToken(record.Group) || !validSource(record.Source) ||
				!utf8.ValidString(record.Prompt) || strings.TrimSpace(record.Prompt) == "" ||
				strings.ContainsAny(record.Prompt, "\r\n") || utf8.RuneCountInString(record.Prompt) > trainingPromptWidth {
				return errors.New("controller training: invalid record")
			}
			if _, duplicate := ids[record.ID]; duplicate {
				return errors.New("controller training: duplicate record")
			}
			ids[record.ID] = struct{}{}
			if splitIndex == 0 {
				trainGroups[record.Group] = struct{}{}
			} else if _, leaked := trainGroups[record.Group]; leaked {
				return errors.New("controller training: group crosses external holdout")
			}
			if err := validateAction(record.Action); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateAction(action Action) error {
	if action.Scope != ScopeComponent && action.Scope != ScopeWorkflow || !validModality(action.Modality) {
		return errors.New("controller training: invalid action scope or modality")
	}
	module, ok := workflowrecipe.Module(action.Module)
	if !ok || !slices.Contains(module.Tasks, action.Task) {
		return errors.New("controller training: action module does not own task")
	}
	expected := map[Modality]recipe.ModuleID{
		ModalityText: workflowrecipe.ModuleGenerate, ModalityImage: workflowrecipe.ModuleProjectImage,
		ModalityAudio: workflowrecipe.ModuleProjectAudio, ModalityVideo: workflowrecipe.ModuleProjectVideo,
	}
	if action.Scope == ScopeComponent && action.Module != expected[action.Modality] {
		return errors.New("controller training: component module differs from modality")
	}
	if action.Scope == ScopeWorkflow && action.Task != recipe.TaskTraining {
		return errors.New("controller training: workflow action is not training orchestration")
	}
	return nil
}

func validateCoverage(actions []Action) error {
	scopes, modalities := map[Scope]bool{}, map[Modality]bool{}
	for _, action := range actions {
		scopes[action.Scope], modalities[action.Modality] = true, true
	}
	if !scopes[ScopeComponent] || !scopes[ScopeWorkflow] ||
		!modalities[ModalityText] || !modalities[ModalityImage] || !modalities[ModalityAudio] || !modalities[ModalityVideo] {
		return errors.New("controller training: component, workflow, and multimodal coverage required")
	}
	return nil
}

func collectActions(parts ...[]Record) []Action {
	byKey := map[string]Action{}
	for _, records := range parts {
		for _, record := range records {
			byKey[actionKey(record.Action)] = record.Action
		}
	}
	keys := make([]string, 0, len(byKey))
	for key := range byKey {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	result := make([]Action, len(keys))
	for index, key := range keys {
		result[index] = byKey[key]
	}
	return result
}

func actionKey(action Action) string {
	return strings.Join([]string{string(action.Scope), string(action.Task), string(action.Modality), string(action.Module)}, "|")
}

func recordIDs(records []Record) []string {
	result := make([]string, len(records))
	for index := range records {
		result[index] = records[index].ID
	}
	return result
}

func validModality(value Modality) bool {
	return value == ModalityText || value == ModalityImage || value == ModalityAudio || value == ModalityVideo
}

func validToken(value string) bool {
	return value != "" && len(value) <= 128 && strings.Trim(value, "abcdefghijklmnopqrstuvwxyz0123456789-_.") == ""
}

func validSource(source GitSource) bool {
	if len(source.Commit) != 40 && len(source.Commit) != 64 || strings.Trim(source.Commit, "0123456789abcdef") != "" {
		return false
	}
	clean := path.Clean(source.Path)
	return source.Path != "" && clean == source.Path && clean != "." && !strings.HasPrefix(clean, "../") && !strings.HasPrefix(clean, "/") && !strings.Contains(source.Path, "\\")
}
