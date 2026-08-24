package server

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

type ResponseFile struct {
	Data      []byte
	Filename  string
	MediaType string
}

type ResponseFileResolver interface {
	ResolveResponseFile(string) (ResponseFile, bool)
}

type ResponseFilePolicy struct {
	Schema        int                        `json:"schema"`
	ResponseFiles responseFilePolicySettings `json:"response_files"`
	files         map[string]ResponseFile
}

type responseFilePolicySettings struct {
	Enabled      bool                          `json:"enabled"`
	MaxFileBytes int64                         `json:"max_file_bytes"`
	AllowedRoots []string                      `json:"allowed_roots"`
	Files        map[string]responseFileRecord `json:"files"`
}

type responseFileRecord struct {
	Path      string `json:"path"`
	MediaType string `json:"media_type"`
}

var responseFileIDPattern = regexp.MustCompile(`^file_[A-Za-z0-9._-]+$`)

func LoadResponseFilePolicy(path string) (*ResponseFilePolicy, error) {
	var policy ResponseFilePolicy
	if err := loadPolicyDocument(path, &policy); err != nil {
		return nil, fmt.Errorf("load response resource policy: %w", err)
	}
	if policy.Schema != policyDocumentSchemaVersion {
		return nil, fmt.Errorf("response resource policy schema %d is unsupported", policy.Schema)
	}
	if !policy.ResponseFiles.Enabled {
		if len(policy.ResponseFiles.Files) != 0 || len(policy.ResponseFiles.AllowedRoots) != 0 {
			return nil, errors.New("disabled response resource policy must not declare roots or files")
		}
		policy.files = map[string]ResponseFile{}
		return &policy, nil
	}
	if policy.ResponseFiles.MaxFileBytes <= 0 {
		return nil, errors.New("response resource max_file_bytes must be positive")
	}
	base, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return nil, err
	}
	roots := make([]string, len(policy.ResponseFiles.AllowedRoots))
	for index, root := range policy.ResponseFiles.AllowedRoots {
		resolved, resolveErr := resolvePolicyPath(base, root)
		if resolveErr != nil {
			return nil, fmt.Errorf("response resource root %d: %w", index, resolveErr)
		}
		roots[index] = resolved
	}
	if len(roots) == 0 && len(policy.ResponseFiles.Files) != 0 {
		return nil, errors.New("response resource files require an allowed root")
	}
	policy.files = make(map[string]ResponseFile, len(policy.ResponseFiles.Files))
	for id, record := range policy.ResponseFiles.Files {
		if !responseFileIDPattern.MatchString(id) {
			return nil, fmt.Errorf("response resource ID %q is invalid", id)
		}
		resolved, resolveErr := resolvePolicyPath(base, record.Path)
		if resolveErr != nil {
			return nil, fmt.Errorf("response resource %q: %w", id, resolveErr)
		}
		if !pathWithinAnyRoot(resolved, roots) {
			return nil, fmt.Errorf("response resource %q is outside allowed roots", id)
		}
		content, readErr := readBoundedFile(resolved, policy.ResponseFiles.MaxFileBytes, true)
		if readErr != nil {
			return nil, fmt.Errorf("response resource %q: %w", id, readErr)
		}
		mediaType := strings.ToLower(strings.TrimSpace(record.MediaType))
		if !supportedResponseFileType(mediaType) {
			return nil, fmt.Errorf("response resource %q media type %q is unsupported", id, mediaType)
		}
		policy.files[id] = ResponseFile{
			Data: content, Filename: filepath.Base(resolved), MediaType: mediaType,
		}
	}
	return &policy, nil
}

func (p *ResponseFilePolicy) ResolveResponseFile(id string) (ResponseFile, bool) {
	if p == nil {
		return ResponseFile{}, false
	}
	file, ok := p.files[id]
	if !ok {
		return ResponseFile{}, false
	}
	file.Data = slices.Clone(file.Data)
	return file, true
}

func resolvePolicyPath(base, path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", errors.New("path is empty")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(base, path)
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return "", err
	}
	return filepath.Clean(resolved), nil
}

func pathWithinAnyRoot(path string, roots []string) bool {
	for _, root := range roots {
		relative, err := filepath.Rel(root, path)
		if err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

func readBoundedFile(path string, maximum int64, rejectEmpty bool) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maximum+1))
	if err != nil {
		return nil, err
	}
	if rejectEmpty && len(data) == 0 {
		return nil, errors.New("file is empty")
	}
	if int64(len(data)) > maximum {
		return nil, errors.New("file exceeds byte limit")
	}
	return data, nil
}

func supportedResponseFileType(mediaType string) bool {
	return strings.HasPrefix(mediaType, "image/") || supportedResponseTextType(mediaType)
}

func supportedResponseTextType(mediaType string) bool {
	return strings.HasPrefix(mediaType, "text/") || mediaType == "application/json" ||
		mediaType == "application/xml"
}
