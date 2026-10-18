package main

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/atomicfile"
	"overgo/internal/clioptions"
	"overgo/internal/hfhub"
	"overgo/internal/hfrepo"
	"overgo/internal/pathidentity"
)

func writeSourceReceipt(ctx context.Context, provider, endpoint string, kind hfhub.RepoKind, resolved hfhub.RepoRevision, directory, output string) error {
	if inside, err := pathidentity.Contains(directory, output); err != nil || inside {
		return fmt.Errorf("hf-hub: receipt %q must be outside the downloaded model directory", output)
	}
	if endpoint == "" {
		endpoint = hfhub.PublicEndpoint
		if provider == "modelscope" {
			endpoint = "https://www.modelscope.cn"
		}
	}
	source := strings.TrimRight(endpoint, "/")
	if kind == hfhub.KindDataset {
		source += "/datasets"
	}
	receipt := hfrepo.SourceReceipt{Source: source + "/" + resolved.ID + ".git", Revision: resolved.Revision}
	for _, file := range resolved.Files {
		receipt.Files = append(receipt.Files, hfrepo.SourceFile{Path: file.Path, SHA256: file.SHA256, Bytes: file.Size})
	}
	verified, err := receipt.Verify(ctx, directory)
	if err != nil {
		return fmt.Errorf("hf-hub: cannot publish a source receipt without complete digest verification: %w", err)
	}
	data, err := json.MarshalIndent(verified, "", "  ")
	if err != nil {
		return err
	}
	return atomicfile.Write(filepath.Clean(output), append(data, '\n'), clioptions.PrivateFileMode)
}
