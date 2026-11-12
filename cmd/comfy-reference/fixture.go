package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/fsatomic"
	"overgo/internal/jsonfile"
)

// ultraFixture is where MOSS Ultra's workflow and ComfyUI's requests for it are kept.
var ultraFixture = filepath.Join("fixtures", "comfy", "moss-ultra")

// The files of a fixture: the workflow as saved, the hash it was saved with, the node definitions it uses, its sections
// by title, and ComfyUI's own request for each section.
const (
	workflowFile   = "workflow.json"
	sourceFile     = "source.json"
	objectInfoFile = "object_info.json"
	sectionsFile   = "sections.json"
	requestsFolder = "requests"
	// savedRequest is the name ComfyUI's page saves a section's request under, before the section's slug.
	savedRequest = "api_prompt_"
	// fixtureMode is a fixture file's mode, read by anyone as the repository's other files are.
	fixtureMode = 0o644
)

// source names the workflow a fixture keeps, the SHA-256 it was kept with, and the file of the requests folder that
// holds ComfyUI's request for each section, by its title, as a path beneath the fixture.
type source struct {
	Workflow string            `json:"workflow"`
	SHA256   string            `json:"sha256"`
	Requests map[string]string `json:"requests"`
}

// section is a group of the workflow's top graph: the nodes a switch turns on together.
type section struct {
	Nodes []string `json:"nodes"`
}

// fixture is a kept workflow with what reading it needs.
type fixture struct {
	folder   string
	workflow []byte
	info     map[string]nodeInfo
	sections map[string]section
	// requests are the files beneath the fixture holding ComfyUI's request for each section, by its title.
	requests map[string]string
}

// slug is a section's title as a file name ("TEXT TO IMAGE" → "text-to-image").
func slug(title string) string {
	return strings.ToLower(strings.Join(strings.Fields(title), "-"))
}

// importFixture keeps workflow with its hash, the definitions of the node types it uses out of objectInfo, and the
// sections and ComfyUI's requests for them that ComfyUI's page saved to saved.
func importFixture(folder, workflowPath, objectInfoPath, saved string) error {
	if objectInfoPath == "" || saved == "" {
		return errors.New("comfy-reference: -import-workflow needs -import-object-info and -import-saved")
	}
	workflow, err := os.ReadFile(workflowPath)
	if err != nil {
		return err
	}
	var parsed uiWorkflow
	if err := json.Unmarshal(workflow, &parsed); err != nil {
		return fmt.Errorf("comfy-reference: %s: %w", workflowPath, err)
	}
	var everything map[string]json.RawMessage
	if err := readJSON(objectInfoPath, &everything); err != nil {
		return err
	}
	used := map[string]json.RawMessage{}
	for _, nodes := range graphs(parsed) {
		for _, node := range nodes {
			if definition, held := everything[node.Type]; held {
				used[node.Type] = definition
			}
		}
	}
	var sections map[string]section
	if err := readJSON(filepath.Join(saved, sectionsFile), &sections); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(folder, requestsFolder), fsatomic.DirectoryMode); err != nil {
		return err
	}
	kept := source{Workflow: filepath.Base(workflowPath), SHA256: hashOf(workflow), Requests: map[string]string{}}
	for title := range sections {
		request, err := os.ReadFile(filepath.Join(saved, savedRequest+slug(title)+".json"))
		if err != nil {
			return fmt.Errorf("comfy-reference: ComfyUI's request for section %q: %w", title, err)
		}
		kept.Requests[title] = path.Join(requestsFolder, slug(title)+".json")
		if err := os.WriteFile(filepath.Join(folder, filepath.FromSlash(kept.Requests[title])), request, fixtureMode); err != nil {
			return err
		}
	}
	if err := os.WriteFile(filepath.Join(folder, workflowFile), workflow, fixtureMode); err != nil {
		return err
	}
	for name, value := range map[string]any{sourceFile: kept, objectInfoFile: used, sectionsFile: sections} {
		if err := jsonfile.Write(filepath.Join(folder, name), value, fixtureMode); err != nil {
			return err
		}
	}
	return nil
}

// graphs are the workflow's top graph's nodes and each subgraph's.
func graphs(workflow uiWorkflow) [][]*uiNode {
	all := [][]*uiNode{workflow.Nodes}
	for _, subgraph := range workflow.Definitions.Subgraphs {
		all = append(all, subgraph.Nodes)
	}
	return all
}

// readFixture reads the fixture in folder, refusing a workflow whose hash is not the one it was kept with.
func readFixture(folder string) (*fixture, error) {
	f := &fixture{folder: folder}
	var kept source
	if err := readJSON(filepath.Join(folder, sourceFile), &kept); err != nil {
		return nil, err
	}
	workflow, err := os.ReadFile(filepath.Join(folder, workflowFile))
	if err != nil {
		return nil, err
	}
	if hashOf(workflow) != kept.SHA256 {
		return nil, fmt.Errorf("comfy-reference: %s is not the %s kept with SHA-256 %s", workflowFile, kept.Workflow, kept.SHA256)
	}
	f.workflow, f.requests = workflow, kept.Requests
	if err := readJSON(filepath.Join(folder, objectInfoFile), &f.info); err != nil {
		return nil, err
	}
	if err := readJSON(filepath.Join(folder, sectionsFile), &f.sections); err != nil {
		return nil, err
	}
	for title := range f.sections {
		if f.requests[title] == "" {
			return nil, fmt.Errorf("comfy-reference: %s names no request of ComfyUI's for section %q", sourceFile, title)
		}
	}
	return f, nil
}

// titles are the fixture's sections in order.
func (f *fixture) titles() []string {
	titles := make([]string, 0, len(f.sections))
	for title := range f.sections {
		titles = append(titles, title)
	}
	slices.Sort(titles)
	return titles
}

// request is the API request of the workflow with the section titled title switched on.
func (f *fixture) request(title string) (map[string]apiNode, error) {
	chosen, held := f.sections[title]
	if !held {
		return nil, fmt.Errorf("comfy-reference: the workflow has no section %q (%s)", title, strings.Join(f.titles(), ", "))
	}
	members := make([]int, 0, len(chosen.Nodes))
	for _, member := range chosen.Nodes {
		id, err := strconv.Atoi(member)
		if err != nil {
			return nil, fmt.Errorf("comfy-reference: section %q names node %q: %w", title, member, err)
		}
		members = append(members, id)
	}
	var workflow uiWorkflow
	if err := json.Unmarshal(f.workflow, &workflow); err != nil {
		return nil, err
	}
	flattener, err := newFlattener(workflow, f.info)
	if err != nil {
		return nil, err
	}
	if err := flattener.switchOn(members); err != nil {
		return nil, err
	}
	return flattener.request()
}

// readJSON reads a JSON file into value, naming the file where it does not parse.
func readJSON(path string, value any) error {
	if err := jsonfile.Decode(path, value); err != nil {
		return fmt.Errorf("comfy-reference: %s: %w", path, err)
	}
	return nil
}
