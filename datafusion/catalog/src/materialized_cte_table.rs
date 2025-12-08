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

//! MaterializedCTETable implementation for materialized CTEs

use std::sync::Arc;
use std::{any::Any, borrow::Cow};

use crate::Session;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;

use datafusion_physical_plan::ExecutionPlan;

use crate::TableProvider;
use datafusion_common::Result;
use datafusion_expr::{Expr, LogicalPlan, TableProviderFilterPushDown, TableType};
use datafusion_physical_plan::cte::{CTEScanExec, MaterializedCTEState};

/// A table provider for materialized CTEs
///
/// This provider returns a CTEScanExec when scanned. The CTEScanExec is initially
/// created with a dummy state (1 partition), and the actual MaterializedCTEState
/// with the correct partition count is injected later by the physical planner
/// when processing the CTE node via assign_cte_state().
///
/// # Architecture
///
/// The physical planner creates:
/// 1. One MaterializedCTEExec that materializes the CTE query
/// 2. Multiple CTEScanExec instances (one per CTE reference) that read from it
/// 3. All CTEScanExec instances are updated via assign_cte_state() to share the
///    same MaterializedCTEState for coordination
#[derive(Debug)]
pub struct MaterializedCTETable {
    /// The name of the CTE
    name: String,
    /// Schema of the CTE result
    table_schema: SchemaRef,
}

impl MaterializedCTETable {
    /// Construct a new MaterializedCTETable with the given name and schema
    pub fn new(name: String, table_schema: SchemaRef) -> Self {
        Self { name, table_schema }
    }

    /// Get the name of this CTE
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the schema of this CTE
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.table_schema)
    }
}

#[async_trait]
impl TableProvider for MaterializedCTETable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_logical_plan(&'_ self) -> Option<Cow<'_, LogicalPlan>> {
        None
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.table_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Create a CTEScanExec with a dummy state (0 partition)
        // The actual state with correct partition count will be injected later
        // by the physical planner when processing the CTE node via assign_cte_state()
        Ok(Arc::new(CTEScanExec::new(
            self.name.clone(),
            Arc::clone(&self.table_schema),
            Arc::new(MaterializedCTEState::new(0)),
        )))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        // Filters are not pushed down to CTE; they are applied by FilterExec on top
        Ok(vec![
            TableProviderFilterPushDown::Unsupported;
            filters.len()
        ])
    }
}
