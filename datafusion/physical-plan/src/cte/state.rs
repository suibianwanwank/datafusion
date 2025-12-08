use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use datafusion_execution::disk_manager::RefCountedTempFile;

use crate::once_async::OnceAsync;

/// Location of the materialized CTE output.
#[derive(Debug)]
pub enum CTEStorage {
    /// All batches stored in memory.
    Memory(Vec<RecordBatch>),

    /// Data spilled to a temporary file.
    Spilled(RefCountedTempFile),
}

/// Handle to a lazily-materialized CTE partition.
pub type MaterializedPartitionFut = Arc<OnceAsync<CTEStorage>>;

/// Shared OnceAsync handles for the partitions of a materialized CTE.
#[derive(Debug)]
pub struct MaterializedCTEState {
    partitions: Vec<MaterializedPartitionFut>,
}

impl MaterializedCTEState {
    pub fn new(num_partitions: usize) -> Self {
        Self {
            partitions: (0..num_partitions)
                .map(|_| Arc::new(OnceAsync::default()))
                .collect(),
        }
    }

    pub fn partitions(&self) -> &[MaterializedPartitionFut] {
        &self.partitions
    }
}
