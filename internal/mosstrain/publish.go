package mosstrain

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/fsatomic"
	"overgo/internal/peft"
)

// transformerDirectory is the component a served model replaces.
const transformerDirectory = "transformer"

// shardIndexSuffix ends a base's shard index, which the merge writes its own of:
// a directory holding two refuses to open.
const shardIndexSuffix = ".safetensors.index.json"

// CheckServedTarget refuses, before a run spends the device, a target PublishServed could not write: one that exists, and
// one a hard link from base cannot reach, as on another volume. Either would otherwise fail at the end of the run, after its
// checkpoint and before its held-out reading. It leaves the target's parent made and the target absent.
func CheckServedTarget(base, target string) error {
	if _, err := os.Lstat(target); err == nil {
		return fmt.Errorf("mosstrain: served target %s exists, and a run publishes a new directory there", target)
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	var source string
	err := filepath.WalkDir(base, func(path string, entry fs.DirEntry, err error) error {
		if err != nil || source != "" || entry.IsDir() {
			return err
		}
		source = path
		return nil
	})
	if err != nil {
		return err
	}
	if source == "" {
		return fmt.Errorf("mosstrain: the base %s holds no file to serve from", base)
	}
	if err := os.MkdirAll(filepath.Dir(target), fsatomic.DirectoryMode); err != nil {
		return err
	}
	if err := os.Link(source, target); err != nil {
		return fmt.Errorf("mosstrain: served target %s shares no volume a hard link from %s reaches: %w", target, base, err)
	}
	return os.Remove(target)
}

// PublishServed writes a model directory that serves adapter merged into base:
// the transformer's weights merged and its configuration copied, and each other
// entry a hard link to the base's, so a trained model costs its transformer's
// bytes and shares the text encoder, the tokenizer and the VAE.
func PublishServed(base, target string, adapter peft.Adapter) error {
	entries, err := os.ReadDir(base)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(target, fsatomic.DirectoryMode); err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.Name() != transformerDirectory {
			if err := linkTree(filepath.Join(base, entry.Name()), filepath.Join(target, entry.Name())); err != nil {
				return err
			}
		}
	}
	source, destination := filepath.Join(base, transformerDirectory), filepath.Join(target, transformerDirectory)
	if _, err := peft.Merge(adapter, source, destination); err != nil {
		return fmt.Errorf("mosstrain: merge the adapter: %w", err)
	}
	components, err := os.ReadDir(source)
	if err != nil {
		return err
	}
	for _, entry := range components {
		if filepath.Ext(entry.Name()) == ".safetensors" || strings.HasSuffix(entry.Name(), shardIndexSuffix) {
			continue
		}
		if _, err := os.Stat(filepath.Join(destination, entry.Name())); err == nil {
			continue
		}
		if err := linkTree(filepath.Join(source, entry.Name()), filepath.Join(destination, entry.Name())); err != nil {
			return err
		}
	}
	return nil
}

// linkTree hard-links a file, or every file under a directory, to target.
func linkTree(source, target string) error {
	info, err := os.Stat(source)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return os.Link(source, target)
	}
	if err := os.MkdirAll(target, fsatomic.DirectoryMode); err != nil {
		return err
	}
	entries, err := os.ReadDir(source)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if err := linkTree(filepath.Join(source, entry.Name()), filepath.Join(target, entry.Name())); err != nil {
			return err
		}
	}
	return nil
}
