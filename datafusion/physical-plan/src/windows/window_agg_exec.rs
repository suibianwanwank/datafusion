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

//! Stream and channel implementations for window function expressions.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::utils::create_schema;
use crate::execution_plan::EmissionType;
use crate::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use crate::windows::{
    calc_requirements, get_ordered_partition_by_indices, get_partition_by_sort_exprs,
    window_equivalence_properties,
};
use crate::{
    ColumnStatistics, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan,
    ExecutionPlanProperties, PhysicalExpr, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream, Statistics, WindowExpr,
};

use arrow::array::ArrayRef;
use arrow::compute::{concat, concat_batches};
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use datafusion_common::stats::Precision;
use datafusion_common::utils::{evaluate_partition_ranges, transpose};
use datafusion_common::{exec_err, internal_err, DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, LexRequirement};

use futures::{ready, Stream, StreamExt};

/// Window execution plan
#[derive(Debug, Clone)]
pub struct WindowAggExec {
    /// Input plan
    pub(crate) input: Arc<dyn ExecutionPlan>,
    /// Window function expression
    window_expr: Vec<Arc<dyn WindowExpr>>,
    /// Schema after the window is run
    schema: SchemaRef,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Partition by indices that defines preset for existing ordering
    // see `get_ordered_partition_by_indices` for more details.
    ordered_partition_by_indices: Vec<usize>,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: PlanProperties,
    /// If `can_partition` is false, partition_keys is always empty.
    can_repartition: bool,
}

impl WindowAggExec {
    /// Create a new execution plan for window aggregates
    pub fn try_new(
        window_expr: Vec<Arc<dyn WindowExpr>>,
        input: Arc<dyn ExecutionPlan>,
        can_repartition: bool,
    ) -> Result<Self> {
        let schema = create_schema(&input.schema(), &window_expr)?;
        let schema = Arc::new(schema);

        let ordered_partition_by_indices =
            get_ordered_partition_by_indices(window_expr[0].partition_by(), &input);
        let cache = Self::compute_properties(Arc::clone(&schema), &input, &window_expr);
        Ok(Self {
            input,
            window_expr,
            schema,
            metrics: ExecutionPlanMetricsSet::new(),
            ordered_partition_by_indices,
            cache,
            can_repartition,
        })
    }

    /// Window expressions
    pub fn window_expr(&self) -> &[Arc<dyn WindowExpr>] {
        &self.window_expr
    }

    /// Input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Return the output sort order of partition keys: For example
    /// OVER(PARTITION BY a, ORDER BY b) -> would give sorting of the column a
    // We are sure that partition by columns are always at the beginning of sort_keys
    // Hence returned `PhysicalSortExpr` corresponding to `PARTITION BY` columns can be used safely
    // to calculate partition separation points
    pub fn partition_by_sort_keys(&self) -> Result<LexOrdering> {
        let partition_by = self.window_expr()[0].partition_by();
        get_partition_by_sort_exprs(
            &self.input,
            partition_by,
            &self.ordered_partition_by_indices,
        )
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        schema: SchemaRef,
        input: &Arc<dyn ExecutionPlan>,
        window_exprs: &[Arc<dyn WindowExpr>],
    ) -> PlanProperties {
        // Calculate equivalence properties:
        let eq_properties = window_equivalence_properties(&schema, input, window_exprs);

        // Get output partitioning:
        // Because we can have repartitioning using the partition keys this
        // would be either 1 or more than 1 depending on the presence of repartitioning.
        let output_partitioning = input.output_partitioning().clone();

        // Construct properties cache:
        PlanProperties::new(
            eq_properties,
            output_partitioning,
            // TODO: Emission type and boundedness information can be enhanced here
            EmissionType::Final,
            input.boundedness(),
        )
    }

    pub fn partition_keys(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        if !self.can_repartition {
            vec![]
        } else {
            let all_partition_keys = self
                .window_expr()
                .iter()
                .map(|expr| expr.partition_by().to_vec())
                .collect::<Vec<_>>();

            all_partition_keys
                .into_iter()
                .min_by_key(|s| s.len())
                .unwrap_or_else(Vec::new)
        }
    }
}

impl DisplayAs for WindowAggExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "WindowAggExec: ")?;
                let g: Vec<String> = self
                    .window_expr
                    .iter()
                    .map(|e| {
                        format!(
                            "{}: {:?}, frame: {:?}",
                            e.name().to_owned(),
                            e.field(),
                            e.get_window_frame()
                        )
                    })
                    .collect();
                write!(f, "wdw=[{}]", g.join(", "))?;
            }
            DisplayFormatType::TreeRender => {
                let g: Vec<String> = self
                    .window_expr
                    .iter()
                    .map(|e| e.name().to_owned().to_string())
                    .collect();
                writeln!(f, "select_list={}", g.join(", "))?;
            }
        }
        Ok(())
    }
}

impl ExecutionPlan for WindowAggExec {
    fn name(&self) -> &'static str {
        "WindowAggExec"
    }

    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn required_input_ordering(&self) -> Vec<Option<LexRequirement>> {
        let partition_bys = self.window_expr()[0].partition_by();
        let order_keys = self.window_expr()[0].order_by();
        if self.ordered_partition_by_indices.len() < partition_bys.len() {
            vec![calc_requirements(partition_bys, order_keys.iter())]
        } else {
            let partition_bys = self
                .ordered_partition_by_indices
                .iter()
                .map(|idx| &partition_bys[*idx]);
            vec![calc_requirements(partition_bys, order_keys.iter())]
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        if self.partition_keys().is_empty() {
            vec![Distribution::SinglePartition]
        } else {
            vec![Distribution::HashPartitioned(self.partition_keys())]
        }
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(WindowAggExec::try_new(
            self.window_expr.clone(),
            Arc::clone(&children[0]),
            true,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let stream = Box::pin(WindowAggStream::new(
            Arc::clone(&self.schema),
            self.window_expr.clone(),
            input,
            BaselineMetrics::new(&self.metrics, partition),
            self.partition_by_sort_keys()?,
            self.ordered_partition_by_indices.clone(),
        )?);
        Ok(stream)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        let input_stat = self.input.statistics()?;
        let win_cols = self.window_expr.len();
        let input_cols = self.input.schema().fields().len();
        // TODO stats: some windowing function will maintain invariants such as min, max...
        let mut column_statistics = Vec::with_capacity(win_cols + input_cols);
        // copy stats of the input to the beginning of the schema.
        column_statistics.extend(input_stat.column_statistics);
        for _ in 0..win_cols {
            column_statistics.push(ColumnStatistics::new_unknown())
        }
        Ok(Statistics {
            num_rows: input_stat.num_rows,
            column_statistics,
            total_byte_size: Precision::Absent,
        })
    }
}

/// Compute the window aggregate columns
fn compute_window_aggregates(
    window_expr: &[Arc<dyn WindowExpr>],
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    window_expr
        .iter()
        .map(|window_expr| window_expr.evaluate(batch))
        .collect()
}

/// stream for window aggregation plan
pub struct WindowAggStream {
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    finished: bool,
    window_expr: Vec<Arc<dyn WindowExpr>>,
    partition_by_sort_keys: LexOrdering,
    baseline_metrics: BaselineMetrics,
    ordered_partition_by_indices: Vec<usize>,
    current_batch: Vec<RecordBatch>,
}

impl WindowAggStream {
    /// Create a new WindowAggStream
    pub fn new(
        schema: SchemaRef,
        window_expr: Vec<Arc<dyn WindowExpr>>,
        input: SendableRecordBatchStream,
        baseline_metrics: BaselineMetrics,
        partition_by_sort_keys: LexOrdering,
        ordered_partition_by_indices: Vec<usize>,
    ) -> Result<Self> {
        // In WindowAggExec all partition by columns should be ordered.
        if window_expr[0].partition_by().len() != ordered_partition_by_indices.len() {
            return internal_err!("All partition by columns should have an ordering");
        }

        Ok(Self {
            schema,
            input,
            finished: false,
            window_expr,
            baseline_metrics,
            partition_by_sort_keys,
            ordered_partition_by_indices,
            current_batch: vec![],
        })
    }

    fn stream_compute_aggregates(
        &mut self,
        batch: &RecordBatch,
    ) -> Result<Option<RecordBatch>> {
        if batch.num_rows() == 0 {
            return Ok(None);
        }

        if self.partition_by_sort_keys.is_empty() {
            self.current_batch.push(batch.clone());
            return Ok(None);
        }

        let evaluate_batch = concat_batches(
            &self.input.schema(),
            self.current_batch.iter().chain(std::iter::once(batch)),
        )?;

        let partition_columns = self
            .ordered_partition_by_indices
            .iter()
            .map(|&idx| {
                self.partition_by_sort_keys[idx].evaluate_to_sort_column(&evaluate_batch)
            })
            .collect::<Result<Vec<_>>>()?;

        let partition_ranges =
            evaluate_partition_ranges(evaluate_batch.num_rows(), &partition_columns)?;

        let last_range = partition_ranges.last().cloned().ok_or_else(|| {
            DataFusionError::Execution("Empty partition ranges".to_string())
        })?;
        self.current_batch = vec![
            evaluate_batch.slice(last_range.start, last_range.end - last_range.start)
        ];

        let result_batches  = partition_ranges
            .iter()
            .take(partition_ranges.len() - 1)
            .map(|range| {
                let sliced = evaluate_batch.slice(range.start, range.end - range.start);
                compute_window_aggregates(&self.window_expr, &sliced)
            })
            .collect::<Result<Vec<_>, DataFusionError>>()?;

        let result = self.build_output_batch(
            &evaluate_batch.slice(0, last_range.start),
            result_batches,
        )?;
        Ok(Some(result))
    }

    fn reduce_batches(&mut self) -> Result<Option<Result<RecordBatch>>> {
        if self.current_batch.is_empty() {
            return Ok(None);
        }
        
        let batch = concat_batches(&self.input.schema(), self.current_batch.iter())?;
        let results = vec![compute_window_aggregates(&self.window_expr, &batch)?];
        Ok(Some(self.build_output_batch(&batch, results)))
    }

    /// Helper to build a `RecordBatch` from original columns and new window columns.
    fn build_output_batch(
        &self,
        original_batch: &RecordBatch,
        window_results: Vec<Vec<ArrayRef>>,
    ) -> Result<RecordBatch> {
        let window_columns = transpose(window_results)
            .into_iter()
            .map(|cols| concat(&cols.iter().map(|a| a.as_ref()).collect::<Vec<_>>()))
            .collect::<Result<Vec<ArrayRef>, ArrowError>>()?;

        let mut combined_columns = original_batch.columns().to_vec();
        combined_columns.extend_from_slice(&window_columns);

        RecordBatch::try_new(Arc::clone(&self.schema), combined_columns)
            .map_err(Into::into)
    }
}

impl Stream for WindowAggStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let poll = self.poll_next_inner(cx);
        self.baseline_metrics.record_poll(poll)
    }
}

impl WindowAggStream {
    #[inline]
    fn poll_next_inner(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            if self.finished {
                return Poll::Ready(None);
            }
            return Poll::Ready(Some(match ready!(self.input.poll_next_unpin(cx)) {
                Some(Ok(batch)) => {
                    let result = self.stream_compute_aggregates(&batch)?;
                    if let Some(rb) = result {
                        return Poll::Ready(Some(Ok(rb)));
                    }
                    continue;
                }
                Some(Err(e)) => Err(e),
                None => {
                    self.finished = true;
                    return Poll::Ready(self.reduce_batches()?);
                }
            }));
        }
    }
}

impl RecordBatchStream for WindowAggStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
