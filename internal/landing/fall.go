package landing

import "sync"

// WatchFall reads the free space of the volume that holds directory as the watch starts, each time the volume
// reports a change, and as it stops, and keeps the greatest fall below where it stood at the start: what a run needed
// free, its own writes and those of whatever wrote beside it, with no clock to sample by. fall reads it so far, and
// stop ends the watch and reads it.
func WatchFall(directory string, free func(string) (uint64, error)) (fall func() uint64, stop func() (uint64, error), err error) {
	start, err := free(directory)
	if err != nil {
		return nil, nil, err
	}
	var mu sync.Mutex
	least := start
	read := func() {
		if now, err := free(directory); err == nil {
			mu.Lock()
			least = min(least, now)
			mu.Unlock()
		}
	}
	fall = func() uint64 {
		mu.Lock()
		defer mu.Unlock()
		return start - least
	}
	end, err := watchChanges(directory, read)
	if err != nil {
		return nil, nil, err
	}
	return fall, func() (uint64, error) {
		ended := end()
		read()
		return fall(), ended
	}, nil
}
