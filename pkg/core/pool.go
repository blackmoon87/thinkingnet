package core

import (
	"fmt"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// MatrixPool provides memory-efficient matrix reuse to reduce allocations.
type MatrixPool struct {
	pools   map[string]*sync.Pool
	mutex   sync.RWMutex
	stats   map[string]*PoolStats
	maxSize int
	enabled bool
}

// PoolStats tracks usage statistics for a pool.
type PoolStats struct {
	Gets    int64
	Puts    int64
	Creates int64
	Hits    int64
	Misses  int64
	mutex   sync.RWMutex
}

// NewMatrixPool creates a new matrix pool.
func NewMatrixPool() *MatrixPool {
	return &MatrixPool{
		pools:   make(map[string]*sync.Pool),
		stats:   make(map[string]*PoolStats),
		maxSize: 1000, // Default max pool size
		enabled: true,
	}
}

// NewMatrixPoolWithConfig creates a new matrix pool with configuration.
func NewMatrixPoolWithConfig(maxSize int, enabled bool) *MatrixPool {
	return &MatrixPool{
		pools:   make(map[string]*sync.Pool),
		stats:   make(map[string]*PoolStats),
		maxSize: maxSize,
		enabled: enabled,
	}
}

// getPoolKey generates a key for the pool based on matrix dimensions.
func (p *MatrixPool) getPoolKey(rows, cols int) string {
	return fmt.Sprintf("%dx%d", rows, cols)
}

// Get retrieves a matrix from the pool or creates a new one.
func (p *MatrixPool) Get(rows, cols int) *mat.Dense {
	if !p.enabled {
		return mat.NewDense(rows, cols, nil)
	}

	key := p.getPoolKey(rows, cols)

	// Update stats
	p.updateStats(key, func(stats *PoolStats) {
		stats.Gets++
	})

	p.mutex.RLock()
	pool, exists := p.pools[key]
	p.mutex.RUnlock()

	if !exists {
		p.mutex.Lock()
		// Double-check after acquiring write lock
		if pool, exists = p.pools[key]; !exists {
			pool = &sync.Pool{
				New: func() any {
					p.updateStats(key, func(stats *PoolStats) {
						stats.Creates++
					})
					return mat.NewDense(rows, cols, nil)
				},
			}
			p.pools[key] = pool
			p.stats[key] = &PoolStats{}
		}
		p.mutex.Unlock()
	}

	matrix := pool.Get().(*mat.Dense)

	// Check if we got a reused matrix or a new one
	if matrix != nil {
		p.updateStats(key, func(stats *PoolStats) {
			stats.Hits++
		})
	} else {
		p.updateStats(key, func(stats *PoolStats) {
			stats.Misses++
		})
		matrix = mat.NewDense(rows, cols, nil)
	}

	// Reset the matrix to zeros for clean reuse
	matrix.Zero()
	return matrix
}

// Put returns a matrix to the pool for reuse.
func (p *MatrixPool) Put(matrix *mat.Dense) {
	if !p.enabled || matrix == nil {
		return
	}

	rows, cols := matrix.Dims()
	key := p.getPoolKey(rows, cols)

	// Update stats
	p.updateStats(key, func(stats *PoolStats) {
		stats.Puts++
	})

	p.mutex.RLock()
	pool, exists := p.pools[key]
	p.mutex.RUnlock()

	if exists {
		pool.Put(matrix)
	}
}

// Clear clears all pools.
func (p *MatrixPool) Clear() {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	p.pools = make(map[string]*sync.Pool)
}

// Stats returns statistics about the pool usage.
func (p *MatrixPool) Stats() map[string]PoolStats {
	p.mutex.RLock()
	defer p.mutex.RUnlock()

	result := make(map[string]PoolStats)
	for key, stats := range p.stats {
		stats.mutex.RLock()
		result[key] = *stats
		stats.mutex.RUnlock()
	}
	return result
}

// updateStats safely updates pool statistics.
func (p *MatrixPool) updateStats(key string, updateFn func(*PoolStats)) {
	p.mutex.RLock()
	stats, exists := p.stats[key]
	p.mutex.RUnlock()

	if !exists {
		p.mutex.Lock()
		if stats, exists = p.stats[key]; !exists {
			stats = &PoolStats{}
			p.stats[key] = stats
		}
		p.mutex.Unlock()
	}

	stats.mutex.Lock()
	updateFn(stats)
	stats.mutex.Unlock()
}

// SetEnabled enables or disables the pool.
func (p *MatrixPool) SetEnabled(enabled bool) {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.enabled = enabled
}

// IsEnabled returns whether the pool is enabled.
func (p *MatrixPool) IsEnabled() bool {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	return p.enabled
}

// Global matrix pool instance
var globalMatrixPool = NewMatrixPool()

// GetMatrix retrieves a matrix from the global pool.
func GetMatrix(rows, cols int) *mat.Dense {
	return globalMatrixPool.Get(rows, cols)
}

// PutMatrix returns a matrix to the global pool.
func PutMatrix(matrix *mat.Dense) {
	globalMatrixPool.Put(matrix)
}

// ClearMatrixPool clears the global matrix pool.
func ClearMatrixPool() {
	globalMatrixPool.Clear()
}

// MatrixPoolStats returns statistics about the global matrix pool.
func MatrixPoolStats() map[string]PoolStats {
	return globalMatrixPool.Stats()
}

// SetMatrixPoolEnabled enables or disables the global matrix pool.
func SetMatrixPoolEnabled(enabled bool) {
	globalMatrixPool.SetEnabled(enabled)
}

// IsMatrixPoolEnabled returns whether the global matrix pool is enabled.
func IsMatrixPoolEnabled() bool {
	return globalMatrixPool.IsEnabled()
}
