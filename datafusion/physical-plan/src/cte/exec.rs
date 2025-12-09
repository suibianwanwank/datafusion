// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Execution plans for materialized common table expressions (CTEs).
//!
//! Materialized CTEs run the CTE query once per partition and let every reference reuse the
//! same output. `MaterializedCTEExec` acts as a container, `CTEScanExec` triggers
//! per-partition materialization on demand, and `CTEStorage` keeps the batches either in
//! memory or on disk.

use std::any::Any;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{internal_err, Result};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use futures::{FutureExt, Stream, StreamExt};

use super::state::{CTEStorage, MaterializedCTEState, MaterializedPartitionFut};
use crate::execution_plan::{Boundedness, EmissionType};
use crate::memory::MemoryStream;
use crate::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use crate::spill::{
    get_record_batch_memory_size, in_progress_spill_file::InProgressSpillFile,
    spill_manager::SpillManager,
};
use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream, Statistics,
};

/// Assigns a MaterializedCTEState to all CTEScanExec nodes in the plan tree
/// that reference the specified CTE name.
///
/// This is similar to `assign_work_table` for RecursiveQueryExec.
fn assign_cte_state(
    plan: Arc<dyn ExecutionPlan>,
    cte_name: &str,
    state: Arc<MaterializedCTEState>,
) -> Result<Arc<dyn ExecutionPlan>> {
    plan.transform_down(|plan| {
        // Check if this is a CTEScanExec with matching name
        if let Some(cte_scan) = plan.as_any().downcast_ref::<CTEScanExec>() {
            if cte_scan.name() == cte_name {
                // Use with_new_state to create a new CTEScanExec with the provided state
                if let Some(new_plan) =
                    plan.with_new_state(Arc::clone(&state) as Arc<dyn Any + Send + Sync>)
                {
                    return Ok(Transformed::yes(new_plan));
                }
            }
        }
        Ok(Transformed::no(plan))
    })
    .data()
}

/// Storage strategy during materialization.
enum MaterializationState {
    /// Buffering batches in memory.
    InMemory(Vec<RecordBatch>),
    /// Spilling batches to disk.
    Spilling(SpillManager, InProgressSpillFile),
}

impl MaterializationState {
    /// Try to buffer a batch in memory. If memory is exhausted, transition to spilling.
    async fn try_buffer_or_spill(
        self,
        batch: RecordBatch,
        reservation: &mut MemoryReservation,
        spill_manager: &SpillManager,
    ) -> Result<Self> {
        match self {
            Self::InMemory(mut batches) => {
                let batch_size = get_record_batch_memory_size(&batch);

                // Try to reserve memory for this batch
                if reservation.try_grow(batch_size).is_ok() {
                    batches.push(batch);
                    return Ok(Self::InMemory(batches));
                }

                // Memory exhausted - transition to spilling
                let spill_manager = spill_manager.clone();
                let mut in_progress = spill_manager.create_in_progress_file("CTE")?;

                // Write all previously buffered batches
                for buffered in batches {
                    in_progress.append_batch(&buffered)?;
                }

                // Write the current batch that triggered spilling
                in_progress.append_batch(&batch)?;

                Ok(Self::Spilling(spill_manager, in_progress))
            }
            Self::Spilling(manager, mut in_progress) => {
                // Already spilling - write directly to disk
                in_progress.append_batch(&batch)?;
                Ok(Self::Spilling(manager, in_progress))
            }
        }
    }

    /// Finalize the materialization and return the appropriate storage.
    fn finalize(self) -> Result<CTEStorage> {
        match self {
            Self::InMemory(batches) => Ok(CTEStorage::Memory(batches)),
            Self::Spilling(_manager, mut in_progress) => {
                let spill_file = in_progress.finish()?.ok_or_else(|| {
                    datafusion_common::DataFusionError::Internal(
                        "Spill file is empty".to_string(),
                    )
                })?;
                Ok(CTEStorage::Spilled(spill_file))
            }
        }
    }
}

/// Collects all batches from a single partition and chooses storage (memory or spilled).
async fn materialize_partition(
    mut input: SendableRecordBatchStream,
    schema: SchemaRef,
    mut reservation: MemoryReservation,
    context: Arc<TaskContext>,
    spill_metrics: crate::metrics::SpillMetrics,
) -> Result<CTEStorage> {
    let mut state = MaterializationState::InMemory(Vec::new());
    let spill_manager =
        SpillManager::new(context.runtime_env(), spill_metrics, Arc::clone(&schema));

    while let Some(batch) = input.next().await.transpose()? {
        state = state
            .try_buffer_or_spill(batch, &mut reservation, &spill_manager)
            .await?;
    }

    state.finalize()
}

/// Execution plan that materializes a CTE and exposes shared state.
/// This plan simply passes through to its input - the actual materialization
/// happens lazily when CTEScanExec partitions are executed.
#[derive(Debug)]
pub struct MaterializedCTEExec {
    /// Name of the CTE (for debugging/metrics)
    name: String,

    /// The CTE query to materialize
    cte_query: Arc<dyn ExecutionPlan>,

    /// The main query that uses the CTE (contains CTEScanExecs)
    input: Arc<dyn ExecutionPlan>,

    /// Shared materialized state (one OnceAsync per partition).
    /// Each partition is materialized independently on first access.
    state: Arc<MaterializedCTEState>,

    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,

    /// Cache holding plan properties
    cache: PlanProperties,
}

impl MaterializedCTEExec {
    /// Create a new MaterializedCTEExec.
    pub fn try_new(
        name: String,
        cte_query: Arc<dyn ExecutionPlan>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        // Traverse the input plan and inject the state into all CTEScanExec
        // nodes that reference this CTE.
        let state = Arc::new(MaterializedCTEState::new(
            cte_query
                .properties()
                .output_partitioning()
                .partition_count(),
        ));
        let input = assign_cte_state(input, &name, Arc::clone(&state))?;

        let eq_properties = input.properties().equivalence_properties().clone();
        let partitioning = input.properties().output_partitioning().clone();
        let cache = PlanProperties::new(
            eq_properties,
            // Preserve the partitioning of the main query input
            partitioning,
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Self {
            name,
            cte_query,
            input,
            state,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }
}

impl DisplayAs for MaterializedCTEExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "MaterializedCTEExec: name={}", self.name)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "name={}", self.name)
            }
        }
    }
}

impl ExecutionPlan for MaterializedCTEExec {
    fn name(&self) -> &'static str {
        "MaterializedCTEExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // First child is the CTE query itself - it doesn't maintain order
        // Second child is the main query (input) - it DOES maintain order
        vec![false, true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![true, true]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.cte_query, &self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 2 {
            return internal_err!(
                "MaterializedCTEExec expected 2 children, got {}",
                children.len()
            );
        }

        Ok(Arc::new(MaterializedCTEExec::try_new(
            self.name.clone(),
            Arc::clone(&children[0]),
            Arc::clone(&children[1]),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Register lazy materialization futures for all CTE partitions (only happens once)
        // Each future spawns a task to materialize its partition only when accessed by CTEScanExec
        for (cte_partition_id, partition_fut) in
            self.state.partitions().iter().enumerate()
        {
            let partition_once = Arc::clone(partition_fut);
            let cte_query = Arc::clone(&self.cte_query);
            let schema = self.cte_query.schema();
            let cte_name = self.name.clone();
            let context_for_mat = Arc::clone(&context);
            let spill_metrics =
                crate::metrics::SpillMetrics::new(&self.metrics, cte_partition_id);

            // Initialize the OnceAsync with a lazy future that will materialize this CTE partition
            // The future is only polled (and spawns the task) when CTEScanExec accesses it
            // try_once ensures this only happens once even if called multiple times
            partition_once.try_once(move || {
                // Return a lazy future that spawns the task only when polled
                Ok(async move {
                    // Spawn a background task to execute CTE materialization
                    // This runs independently to avoid blocking the main query
                    let task =
                        datafusion_common_runtime::SpawnedTask::spawn(async move {
                            // Execute this partition of the CTE query
                            let stream = cte_query
                                .execute(cte_partition_id, Arc::clone(&context_for_mat))?;

                            // Create per-partition memory reservation
                            let reservation = MemoryConsumer::new(format!(
                                "MaterializedCTE[{cte_name}]:partition[{cte_partition_id}]"
                            ))
                            .with_can_spill(true)
                            .register(context_for_mat.memory_pool());

                            // Materialize this partition
                            materialize_partition(
                                stream,
                                schema,
                                reservation,
                                context_for_mat,
                                spill_metrics,
                            )
                            .await
                        });

                    // Wait for the spawned task to complete
                    task.join_unwind().await.map_err(|e| {
                        datafusion_common::DataFusionError::Execution(format!(
                            "CTE materialization task failed: {e}"
                        ))
                    })?
                }
                .boxed())
            })?;
        }

        // Simply pass through to the input plan
        // The actual materialization will be awaited by CTEScanExec
        self.input.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        self.input.partition_statistics(None)
    }
}

/// Execution plan that reads from a materialized CTE.
/// When execute() is called for a partition, it awaits the materialization future
/// that was created by MaterializedCTEExec and returns the cached data.
#[derive(Debug)]
pub struct CTEScanExec {
    /// Name of the CTE being scanned
    name: String,

    /// Schema of the CTE data
    schema: SchemaRef,

    /// Shared partition data (same instances as in MaterializedCTEExec)
    /// Each partition has its own OnceAsync containing the materialization future
    state: Arc<MaterializedCTEState>,

    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,

    /// Cache holding plan properties
    cache: PlanProperties,
}

impl CTEScanExec {
    /// Create a new CTEScanExec.
    pub fn new(
        name: String,
        schema: SchemaRef,
        state: Arc<MaterializedCTEState>,
    ) -> Self {
        let num_partitions = state.partitions().len();
        let cache = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(num_partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            name,
            schema,
            state,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        }
    }

    /// Name of the CTE.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl DisplayAs for CTEScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "CTEScanExec: name={}", self.name)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "name={}", self.name)
            }
        }
    }
}

impl ExecutionPlan for CTEScanExec {
    fn name(&self) -> &'static str {
        "CTEScanExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition >= self.state.partitions().len() {
            return internal_err!(
                "CTEScanExec got invalid partition {} (expected 0..{})",
                partition,
                self.state.partitions().len()
            );
        }

        let partitions = self.state.partitions();
        let partition_fut = Arc::clone(&partitions[partition]);
        let schema = Arc::clone(&self.schema);

        let stream = CTEScanStream::new(schema, partition_fut, context, partition);

        Ok(Box::pin(stream))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.schema))
    }

    /// Creates a new CTEScanExec with the provided shared state.
    ///
    /// This method is used by [`MaterializedCTEExec`] to inject the shared
    /// [`MaterializedCTEState`] into all [`CTEScanExec`] nodes that reference
    /// this CTE. This enables multiple scan nodes to access the same materialized
    /// partitions without re-executing the CTE query.
    ///
    /// # Arguments
    /// * `state` - The shared state containing materialization futures for all partitions
    ///
    /// # Returns
    /// * `Some(Arc<dyn ExecutionPlan>)` if the state can be successfully downcast
    /// * `None` if the state is not a [`MaterializedCTEState`]
    fn with_new_state(
        &self,
        state: Arc<dyn Any + Send + Sync>,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        // Try to downcast to MaterializedCTEState
        let cte_state = state.downcast::<MaterializedCTEState>().ok()?;

        // Create a new CTEScanExec with the provided state
        Some(Arc::new(CTEScanExec::new(
            self.name.clone(),
            Arc::clone(&self.schema),
            cte_state,
        )))
    }
}

/// Stream that reads from a materialized CTE partition.
///
/// This stream waits for the CTE partition to be materialized (via OnceAsync),
/// then delegates to the appropriate underlying stream based on storage type.
struct CTEScanStream {
    /// Schema of the output
    schema: SchemaRef,

    /// Future that produces the materialized partition data
    partition_fut: MaterializedPartitionFut,

    /// TaskContext for creating spill manager
    context: Arc<TaskContext>,

    /// Partition ID for metrics
    partition: usize,

    /// Current state
    state: CTEScanState,
}

enum CTEScanState {
    /// Waiting for materialization to complete.
    WaitingForMaterialization(Option<crate::once_async::OnceFut<CTEStorage>>),

    /// Materialization complete, streaming data
    Streaming(SendableRecordBatchStream),

    /// Stream exhausted
    Done,
}

impl CTEScanStream {
    fn new(
        schema: SchemaRef,
        partition_fut: MaterializedPartitionFut,
        context: Arc<TaskContext>,
        partition: usize,
    ) -> Self {
        Self {
            schema,
            partition_fut,
            context,
            partition,
            state: CTEScanState::WaitingForMaterialization(None),
        }
    }

    /// Poll the materialization future and transition to streaming state
    fn poll_next_inner(
        &mut self,
        cx: &mut Context<'_>,
        once_fut_opt: &mut Option<crate::once_async::OnceFut<CTEStorage>>,
    ) -> Poll<Result<()>> {
        // Get or create the OnceFut, reusing it across polls to preserve waker registration
        let once_fut = once_fut_opt.get_or_insert_with(|| {
            self.partition_fut.try_once(|| {
            let partition = self.partition;
            Ok(async move {
                internal_err!(
                    "CTEScanExec partition {} accessed before MaterializedCTEExec initialized it",
                    partition
                )
            }
            .boxed())
            }).unwrap()
        });

        let partition_data = std::task::ready!(once_fut.get_shared(cx))?;
        let stream = match &*partition_data {
            CTEStorage::Memory(batches) => {
                let memory_stream = MemoryStream::try_new(
                    batches.clone(),
                    Arc::clone(&self.schema),
                    None, // no projection
                )?;
                Box::pin(memory_stream)
            }
            CTEStorage::Spilled(spill_file) => {
                let file = spill_file.clone();
                let spill_metrics = crate::metrics::SpillMetrics::new(
                    &ExecutionPlanMetricsSet::new(),
                    self.partition,
                );
                let spill_manager = SpillManager::new(
                    self.context.runtime_env(),
                    spill_metrics,
                    Arc::clone(&self.schema),
                );

                spill_manager.read_spill_as_stream(file, None)?
            }
        };

        self.state = CTEScanState::Streaming(stream);
        Poll::Ready(Ok(()))
    }
}

impl Stream for CTEScanStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            // First, check if we're in WaitingForMaterialization state and extract once_fut_opt
            let should_poll_inner =
                matches!(&self.state, CTEScanState::WaitingForMaterialization(_));

            if should_poll_inner {
                // Temporarily take ownership of the state to avoid borrow checker issues
                let old_state = std::mem::replace(&mut self.state, CTEScanState::Done);

                if let CTEScanState::WaitingForMaterialization(mut once_fut_opt) =
                    old_state
                {
                    match self.poll_next_inner(cx, &mut once_fut_opt) {
                        Poll::Ready(Ok(())) => {
                            // Don't restore state, poll_next_inner already transitioned to Streaming
                            continue;
                        }
                        Poll::Ready(Err(e)) => {
                            self.state = CTEScanState::Done;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Pending => {
                            // Restore the state with the (potentially updated) once_fut_opt
                            self.state =
                                CTEScanState::WaitingForMaterialization(once_fut_opt);
                            return Poll::Pending;
                        }
                    }
                }
            }

            // Handle other states
            match &mut self.state {
                CTEScanState::WaitingForMaterialization(_) => {
                    // Should not reach here due to the check above
                    unreachable!("Already handled WaitingForMaterialization");
                }
                CTEScanState::Streaming(stream) => {
                    // Delegate to the underlying stream
                    return match Pin::new(stream).poll_next(cx) {
                        Poll::Ready(Some(batch)) => Poll::Ready(Some(batch)),
                        Poll::Ready(None) => {
                            self.state = CTEScanState::Done;
                            Poll::Ready(None)
                        }
                        Poll::Pending => Poll::Pending,
                    };
                }
                CTEScanState::Done => return Poll::Ready(None),
            }
        }
    }
}

impl RecordBatchStream for CTEScanStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::TestMemoryExec;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::assert_batches_eq;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]))
    }

    fn create_test_batches(
        num_batches: usize,
        rows_per_batch: usize,
    ) -> Vec<RecordBatch> {
        let schema = create_test_schema();
        (0..num_batches)
            .map(|batch_idx| {
                let start = (batch_idx * rows_per_batch) as i32;
                let end = start + rows_per_batch as i32;

                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int32Array::from_iter(start..end)),
                        Arc::new(Int32Array::from_iter((start..end).map(|x| x * 2))),
                    ],
                )
                .unwrap()
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_cte_memory_storage() -> Result<()> {
        let schema = create_test_schema();
        let batches = create_test_batches(3, 5); // 3 batches, 5 rows each

        // Create CTE query (TestMemoryExec with our batches)
        let cte_query =
            TestMemoryExec::try_new(&[batches.clone()], schema.clone(), None)?;
        let cte_query = Arc::new(TestMemoryExec::update_cache(Arc::new(cte_query)));

        // Create CTEScanExec (state will be injected by MaterializedCTEExec)
        let scan = CTEScanExec::new(
            "test_cte".to_string(),
            schema.clone(),
            Arc::new(MaterializedCTEState::new(1)),
        );

        // Create MaterializedCTEExec with scan as input
        // MaterializedCTEExec will create its own state and inject it into the scan
        let exec = MaterializedCTEExec::try_new(
            "test_cte".to_string(),
            cte_query,
            Arc::new(scan),
        )?;

        let context = Arc::new(TaskContext::default());

        // Execute MaterializedCTEExec (which will execute CTEScanExec)
        let mut exec_stream = exec.execute(0, Arc::clone(&context))?;

        // Collect results
        let mut scan_batches = vec![];
        while let Some(batch) = exec_stream.next().await {
            scan_batches.push(batch?);
        }

        // Verify we got the right data
        assert_eq!(scan_batches.len(), 3);

        let expected = [
            "+----+----+",
            "| a  | b  |",
            "+----+----+",
            "| 0  | 0  |",
            "| 1  | 2  |",
            "| 2  | 4  |",
            "| 3  | 6  |",
            "| 4  | 8  |",
            "| 5  | 10 |",
            "| 6  | 12 |",
            "| 7  | 14 |",
            "| 8  | 16 |",
            "| 9  | 18 |",
            "| 10 | 20 |",
            "| 11 | 22 |",
            "| 12 | 24 |",
            "| 13 | 26 |",
            "| 14 | 28 |",
            "+----+----+",
        ];

        assert_batches_eq!(expected, &scan_batches);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_cte_with_spilling() -> Result<()> {
        // Test spilling by using a very small memory limit
        let schema = create_test_schema();
        let batches = create_test_batches(10, 100); // 10 batches, 100 rows each = 1000 rows

        let cte_query =
            TestMemoryExec::try_new(&[batches.clone()], schema.clone(), None)?;
        let cte_query = Arc::new(TestMemoryExec::update_cache(Arc::new(cte_query)));

        // Create runtime with very low memory limit to force spilling
        let runtime = RuntimeEnvBuilder::default()
            .with_memory_limit(1, 1.0) // 1 byte - will definitely spill
            .build_arc()?;

        let context = TaskContext::default().with_runtime(runtime);
        let context = Arc::new(context);

        let scan = CTEScanExec::new(
            "test_cte".to_string(),
            schema.clone(),
            Arc::new(MaterializedCTEState::new(1)),
        );

        let exec = MaterializedCTEExec::try_new(
            "test_cte".to_string(),
            cte_query,
            Arc::new(scan),
        )?;

        // Execute (should trigger spilling via MaterializedCTEExec)
        let mut exec_stream = exec.execute(0, Arc::clone(&context))?;
        let mut scan_batches = vec![];
        let mut total_rows = 0;
        while let Some(batch) = exec_stream.next().await {
            let batch = batch?;
            total_rows += batch.num_rows();
            scan_batches.push(batch);
        }

        // Verify we got all 1000 rows
        assert_eq!(total_rows, 1000);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_cte_lazy_partition_execution() -> Result<()> {
        // Test that partitions are materialized on-demand, not eagerly
        let schema = create_test_schema();

        // Create 3 partitions with distinct data
        let batches_p0 = create_test_batches(1, 10); // partition 0: rows 0-9
        let batches_p1 = create_test_batches(1, 10); // partition 1: rows 0-9
        let batches_p2 = create_test_batches(1, 10); // partition 2: rows 0-9

        // Create a CTE query with 3 partitions
        let cte_query = TestMemoryExec::try_new(
            &[batches_p0.clone(), batches_p1.clone(), batches_p2.clone()],
            schema.clone(),
            None,
        )?;
        let cte_query = Arc::new(TestMemoryExec::update_cache(Arc::new(cte_query)));

        // Create a scan that references the CTE
        let scan = CTEScanExec::new(
            "test_cte".to_string(),
            schema.clone(),
            Arc::new(MaterializedCTEState::new(3)),
        );

        // Create MaterializedCTEExec
        let exec = MaterializedCTEExec::try_new(
            "test_cte".to_string(),
            cte_query,
            Arc::new(scan),
        )?;

        let context = Arc::new(TaskContext::default());

        // Only access partition 1 (not 0 or 2)
        // This verifies that partitions are materialized lazily
        let mut stream = exec.execute(1, Arc::clone(&context))?;
        let mut batches = vec![];
        while let Some(batch) = stream.next().await {
            batches.push(batch?);
        }

        // Verify we got data from partition 1
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 10);
        Ok(())
    }
}
