//go:build !windows

package modeldevice

import "os"

// openShardFile opens a shard through the file cache.
func openShardFile(path string) (shardFile, uint64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	return readerAtFile{source: file, owned: file}, 1, nil
}
