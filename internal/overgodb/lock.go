package overgodb

import "overgo/internal/processlock"

type fileLock = processlock.Lock

func acquireFileLock(path string) (*fileLock, error) {
	return processlock.Acquire(path, storeFileMode)
}
